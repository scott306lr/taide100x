import logging
import torch
import torch.nn as nn
from transformers.generation.logits_process import LogitsWarper, LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, LogitNormalization
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, MaxLengthCriteria, MaxTimeCriteria, EosTokenCriteria


# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
# Several functions are form class GenerationMixin, simplified.
class WrapperBase(nn.Module):
    def __init__(self):
        super(WrapperBase, self).__init__()
        
    # calling .config is same as calling .llm.config
    @property
    def config(self):
        return self.llm.config
    
    def set_llm(self, llm):
        self.llm = llm
        
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        
    def _get_logits_warper(
        self,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ):
        """
        Simplified HuggingFace's `LogitsProcessorList` for multinomial sampling.
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """
        # instantiate warpers list
        warpers = LogitsProcessorList()
        
        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k))
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p))
        
        return warpers
    
    def _get_stopping_criteria(
        self,
        input_ids_length: torch.LongTensor = None,
        max_new_tokens: int = None,
        max_length: int = None,
        max_time: float = None,
        eos_token_tensor: torch.LongTensor = None,
    ):
        criteria = StoppingCriteriaList()
        if max_new_tokens is not None:
            if max_length is not None:
                logging.warning(
                    f"Both `max_new_tokens` (={max_new_tokens}) and `max_length`(="
                    f"{max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                )
            max_length = input_ids_length + max_new_tokens
            
        if max_length is not None:
            max_position_embeddings = getattr(self.llm.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=max_time))
        if eos_token_tensor is not None:
            # EosTokenCriteria only checks last input token,
            # make sure not token is appended after eos_token_tensor during generation
            criteria.append(EosTokenCriteria(eos_token_id=eos_token_tensor))
        return criteria
    
    def _sample_token(
        self,
        logits: torch.FloatTensor,
        logits_warper: LogitsWarper,
        do_sample: bool,
        return_probs: bool = False,
    ):
        if do_sample:
            batch, seq_len, vocab_size = logits.shape
            
            logits = logits.view(-1, vocab_size)
            next_token_scores = logits_warper(None, logits)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            
            if return_probs:
                return probs.view(batch, seq_len, vocab_size) # preserve shape
            else:
                token = torch.multinomial(probs, 1)
                return token.view(batch, seq_len) # preserve shape
        else:
            if return_probs:
                return torch.softmax(logits, dim=-1)
            else:
                return torch.argmax(logits, dim=-1)

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_warper: LogitsWarper,
        do_sample: bool,
        *args,
        **kwargs,
    ):
        r"""
        This method is expected to be implemented by subclasses.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=2048,
        max_length=None,
        do_sample=True,
    ):        
        # 1. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            input_ids_length=input_ids.shape[1],
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            eos_token_tensor=self.tokenizer.eos_token_id
        )
        
        # 2. prepare logits warper (if `do_sample` is `True`)
        logits_warper = (
            self._get_logits_warper(
                temperature=temperature, 
                top_p=top_p, 
                top_k=top_k,
            ) if do_sample else None
        )
        
        # 3. generate
        results = self._generate(
            input_ids=input_ids,
            stopping_criteria=stopping_criteria,
            logits_warper=logits_warper,
            do_sample=do_sample,
        )
        return results