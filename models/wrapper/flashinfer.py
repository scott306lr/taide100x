import torch
import os
from .base import WrapperBase

from transformers.generation.logits_process import LogitsWarper
from transformers.generation.stopping_criteria import StoppingCriteria

from ..llm.utils.flashinfer.cache_manager import (
    KvCachePool,
    KvCacheBatchPosition,
    RequestKvCache,
    getKvCacheBatchPosition,
)

class FlashInferCache():
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self,config) -> None:
        
        currentDevice = torch.device(f'cuda:{torch.cuda.current_device()}')
        PAGE_LEN: int = 16
        dtype_size = torch.tensor([], dtype=torch.float16).element_size()
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        MEMORY_FRACTION = float(os.getenv("CUDA_MEMORY_FRACTION", "1.0"))
        
        cache_page_size = (
                    2   * PAGE_LEN
                        * config.num_hidden_layers
                        * config.num_attention_heads
                        * head_dim
                        * dtype_size
        )

        total_free_memory, _ = torch.cuda.mem_get_info(currentDevice)
        total_gpu_memory = torch.cuda.get_device_properties(currentDevice).total_memory
        free_memory = max(0, total_free_memory - (1 - MEMORY_FRACTION) * total_gpu_memory)    
        num_pages_to_allocate = int(free_memory * 0.80 / cache_page_size)

        self.kvCachePool = KvCachePool(
                max_pages = num_pages_to_allocate,
                num_layers = config.num_hidden_layers,
                num_heads = config.num_attention_heads,
                head_dim = head_dim,
                page_len=PAGE_LEN,
                dtype=torch.float16,
                device=currentDevice,
        )
        
class FlashInferWrapper(WrapperBase):
    def __init__(self):
        super(FlashInferWrapper, self).__init__()
        
   
    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_warper: LogitsWarper,
        do_sample: bool,
    ):
        assert self.llm is not None, "LLM model must be provided"

        # * clone input_ids 
        input_ids = input_ids.clone()

        # * prepare kv-cache
        kvCachePool = FlashInferCache(self.llm.config).kvCachePool
        
        # Get the initial sequence length
        seq_init_len = input_ids.shape[1]
        PAGE_LEN: int = 16
        currentDevice = torch.device(f'cuda:{torch.cuda.current_device()}')

        # Create a RequestKvCache instance
        request_kv_cache = RequestKvCache(
            kvCachePool=kvCachePool,
            page_len=PAGE_LEN,
            seq_init_len=seq_init_len
        )

        # Generate the KvCacheBatchPosition
        batch_position = getKvCacheBatchPosition(
            request_kv_caches=[request_kv_cache],
            isPrefill=True,  # Set to False if you're doing incremental decoding
            device=currentDevice
        )
        
        # * prefill stage
        # outputs = self.llm(input_ids, past_key_values=llm_past_key_values, return_dict=True)
        outputs = self.llm(
            input_ids=input_ids,
            kvCachePool=kvCachePool,
            batch_position=batch_position,
            is_prefill=True,  # Should match the isPrefill used in getKvCacheBatchPosition
            return_dict=True
        )
        
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1:].clone() #TODO: check shape, hf uses outputs.logits[:, -1, :].clone()

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs
        
        next_tokens = self._sample_token(next_token_logits, logits_warper, do_sample)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        
        finished = False
        while not finished:
            
            # Update the KvCacheBatchPosition
            request_kv_cache.increment()
            batch_position = getKvCacheBatchPosition(
                request_kv_caches=[request_kv_cache],
                isPrefill=False,
                device=currentDevice
            )
        
            outputs = self.llm(
                input_ids=input_ids[:, -1:],
                kvCachePool=kvCachePool,
                batch_position=batch_position,
                is_prefill=False,
                return_dict=True
            )
            
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()
            
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
            
            next_tokens = self._sample_token(next_token_logits, logits_warper, do_sample)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # * check stopping criteria
            finished = stopping_criteria(input_ids, None)
            
        return input_ids