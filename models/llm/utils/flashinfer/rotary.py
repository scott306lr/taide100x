from typing import Tuple
import rotary_emb  # flash attention's rotary implementation
import torch
from torch import nn


# cos: seq_len x 1 x rotary_dim
# sin: seq_len x 1 x rotary_dim
# query: seq_len x num_attention_heads x head_dim
# key: seq_len x num_kv_heads x head_dim
# is_neox: whether to use GPT-NeoX's rotary implementation OR GPT-J's rotary implementation
def rotate_query_key_in_place(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox: bool,
):
    rotate_in_place(query, cos, sin, is_neox)
    rotate_in_place(key, cos, sin, is_neox)


def rotate_in_place(x, cos, sin, is_neox):
    rotary_dim = cos.shape[-1] * 2
    if is_neox:
        x1 = x[..., : rotary_dim // 2]
        x2 = x[..., rotary_dim // 2 : rotary_dim]
        rotary_emb.apply_rotary(x1, x2, cos, sin, x1, x2, False)
    else:
        even_positions = list(range(0, rotary_dim, 2))
        odd_positions = list(range(1, rotary_dim, 2))
        x1 = x[..., even_positions]
        x2 = x[..., odd_positions]
        rotary_emb.apply_rotary(x1, x2, cos, sin, x1, x2, False)
        x[..., :rotary_dim] = torch.stack((x1, x2), dim=-1).flatten(start_dim=-2)

def getPositionIdsAndMaxSeqLenForPrefill(
    seq_lens: torch.Tensor, device
) -> Tuple[torch.Tensor, int]:
    if seq_lens.numel() == 0:
        return torch.tensor([], dtype=torch.int32, device=device), 0
    position_ids = torch.cat(
        [
            torch.arange(seq_len, dtype=torch.int32, device=device)
            for seq_len in seq_lens
        ]
    )
    max_seq_len = torch.max(seq_lens).item()
    return position_ids, max_seq_len


def getPositionIdsAndMaxSeqLenForDecode(
    seq_lens: torch.Tensor, device
) -> Tuple[torch.Tensor, int]:
    if seq_lens.numel() == 0:
        return torch.tensor([], dtype=torch.int32, device=device), 0
    position_ids = torch.cat(
        [
            torch.tensor([seq_len - 1], dtype=torch.int32, device=device)
            for seq_len in seq_lens
        ]
    )
    max_seq_len = torch.max(seq_lens).item()
    return position_ids, max_seq_len


# def get_cos_sin(rotary_emb, seq_lens: torch.Tensor, device, dtype, is_prefill):
#     position_ids, max_seq_len = (
#         getPositionIdsAndMaxSeqLenForPrefill(seq_lens, device)
#         if is_prefill
#         else getPositionIdsAndMaxSeqLenForDecode(seq_lens, device)
#     )

#     return rotary_emb.get_cos_sin(position_ids, max_seq_len, dtype)

class PositionRotaryEmbedding(nn.Module):
    def __init__(self, inv_freq, scaling_factor):
        super().__init__()
        self.inv_freq = inv_freq
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.scaling_factor = scaling_factor
        self.dynamic_args = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        # Such controlflows may add some overhead.
        
        rotary_dim = cos.shape[-1]
        q1 = query[..., :rotary_dim]
        q2 = query[..., rotary_dim : 2 * rotary_dim]

        rotary_emb.apply_rotary(q1, q2, cos, sin, q1, q2, False)

        k1 = key[..., :rotary_dim]
        k2 = key[..., rotary_dim : 2 * rotary_dim]

        rotary_emb.apply_rotary(k1, k2, cos, sin, k1, k2, False)
        

    @classmethod
    def static(cls, config, dim, base, device):
        inv_freq = _create_inv_freq(dim, base, device)
        scaling_factor = None
        # rope_scaling = _get_rope_config(config)
        return cls(inv_freq, scaling_factor)
    
    def _update_cos_sin_cache(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            if self.scaling_factor is not None:
                t /= self.scaling_factor
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)

            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def get_cos_sin(self, position_ids: torch.Tensor, max_s: int, dtype: torch.dtype):
        """
        Return cos and sin for the asked position ids
        """
        if SYSTEM == "rocm":
            # For RoCm, we always use float cos/sin to avoid a cast.
            # For NVIDIA, for some reason, the flash-attn rotary kernel requires cos/sin and query/key to be of same dtype: https://github.com/Dao-AILab/flash-attention/blob/017716451d446e464dde9aca3a3c1ed2209caaa9/csrc/rotary/rotary.cpp#L26
            # But later on goes and cast cos/sin to float anyway: https://github.com/Dao-AILab/flash-attention/blob/017716451d446e464dde9aca3a3c1ed2209caaa9/csrc/rotary/rotary_cuda.cu#L29, which looks suboptimal.
            dtype = torch.float32

        self._update_cos_sin_cache(dtype, position_ids.device, max_s)

        cos = torch.index_select(self._cos_cached, 0, position_ids)
        sin = torch.index_select(self._sin_cached, 0, position_ids)

        # Note: this unsqueeze is not necessary on RoCm + VLLM ROPE implementation, but we leave it as is to avoid yet an other controlflow.
        return cos.unsqueeze(1), sin.unsqueeze(1)