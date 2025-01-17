from transformers import AutoTokenizer, BartForConditionalGeneration
import torch.nn as nn
from dataclasses import dataclass
from transformers.utils import logging
from torch.quantization import quantize_dynamic
import torch
@dataclass
class TokenizerConfig:
    bos_token: str
    eos_token: str
    decoder_max_len: int
    encoder_max_len: int
    special_tokens: dict

@dataclass
class DialogueSummarizerConfig:
    model_name: str
    tokenizer: TokenizerConfig

class BartSummarizer(nn.Module):
    def __init__(self, config: DialogueSummarizerConfig):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(config.model_name)
        
        # Add special tokens
        special_tokens_dict = {'additional_special_tokens': [str(token) for token in config.tokenizer.special_tokens.additional_special_tokens]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # 양자화 설정이 활성화된 경우
        if hasattr(config, 'quantization') and config.quantization.enabled:
            self.quantize_model(config.quantization.bits)

    def quantize_model(self, bits=8):
        """모델 양자화"""
        try:
            # Dynamic Quantization
            self.model = quantize_dynamic(
                self.model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            logging.info(f"Model quantized to {bits} bits")
        except Exception as e:
            logging.warning(f"Quantization failed: {str(e)}")

    def forward(self, **inputs):
        return self.model(**inputs) 