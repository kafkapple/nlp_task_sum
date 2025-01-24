from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseModel(ABC):
    @abstractmethod
    def forward(self, **inputs):
        pass
        
    @abstractmethod
    def generate(self, **inputs):
        pass

    def _add_special_tokens(self, tokenizer, special_tokens):
        """특수 토큰 추가를 위한 공통 메서드"""
        if special_tokens:
            special_tokens_dict = {
                'additional_special_tokens': [str(token) for token in special_tokens]
            }
            num_added = tokenizer.add_special_tokens(special_tokens_dict)
            if num_added > 0:
                self.model.resize_token_embeddings(len(tokenizer))
            return num_added
        return 0 