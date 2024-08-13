import torch
from .base import WrapperBase

class HuggingFaceWrapper(WrapperBase):
    def __init__(self):
        super(HuggingFaceWrapper, self).__init__()
    
    def generate(
        self, 
        input_ids: torch.LongTensor, 
        temperature=None, top_p=None, top_k=None, 
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