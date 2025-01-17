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
    def __init__(self, config):
        super().__init__()
        self.config = config.model  # 모델 관련 설정
        self.full_config = config   # 전체 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 토크나이저와 모델 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.name,
            cache_dir=self.full_config.general.model_cache_dir
        )
        self.model = BartForConditionalGeneration.from_pretrained(
            self.config.name,
            cache_dir=self.full_config.general.model_cache_dir
        ).to(self.device)
        
        # Add special tokens (if configured)
        if hasattr(self.config, 'tokenizer') and hasattr(self.config.tokenizer, 'special_tokens'):
            special_tokens_dict = {
                'additional_special_tokens': [
                    "#Person1#", "#Person2#", "#Person3#",
                    "#PhoneNumber#", "#Address#", "#PassportNumber#"
                ]
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))

        # 양자화 설정이 활성화된 경우
        if hasattr(self.config, 'quantization') and self.config.quantization.enabled:
            self.quantize_model()

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