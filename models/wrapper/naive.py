import torch
import torch.nn as nn
from .base import WrapperBase

from transformers.generation.logits_process import LogitsWarper
from transformers.generation.stopping_criteria import StoppingCriteria

from transformers.cache_utils import StaticCache, DynamicCache

class NaiveWrapper(WrapperBase):
    def __init__(self):
        super(NaiveWrapper, self).__init__()
 
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
        llm_past_key_values = DynamicCache()
        
        # * prefill stage
        outputs = self.llm(input_ids, past_key_values=llm_past_key_values, return_dict=True)
        
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
            outputs = self.llm(input_ids[:, -1:], past_key_values=llm_past_key_values, return_dict=True)
        
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