import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsWarper, LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, LogitNormalization
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, MaxLengthCriteria, MaxTimeCriteria, EosTokenCriteria

from transformers.cache_utils import StaticCache, DynamicCache

# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
# Several functions are form class GenerationMixin, simplified.
class SimpleWrapper(nn.Module):
    def __init__(self):
        super(SimpleWrapper, self).__init__()
    
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
        
        return None
    
    def _get_stopping_criteria(
        self,
        max_length: int = None,
        max_time: float = None,
        eos_token_tensor: torch.LongTensor = None,
    ):
        criteria = StoppingCriteriaList()
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
    ):
        if do_sample:
            next_token_scores = logits_warper(None, logits)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(1)
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
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_length=2048,
        do_sample=True,
    ):        
        # 1. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(max_length=max_length, eos_token_tensor=self.tokenizer.eos_token_id)
        
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

class HuggingFaceWrapper(SimpleWrapper):
    def __init__(self):
        super(HuggingFaceWrapper, self).__init__()
    
    def generate(
        self, 
        input_ids: torch.LongTensor, 
        temperature=0, top_p=0, top_k=0, 
        max_length=2048, do_sample=True, 
        *args, 
        **kwargs
    ):
        assert self.llm is not None, "LLM model must be provided"
        
        return self.llm.generate(
            input_ids=input_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_length=max_length,
            do_sample=do_sample,
            *args,
            **kwargs,
        )
        
class NaiveWrapper(SimpleWrapper):
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