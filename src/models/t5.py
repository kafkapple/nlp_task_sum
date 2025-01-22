from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch.nn as nn
from dataclasses import dataclass
from transformers.utils import logging
import torch
from typing import List
from tqdm import tqdm

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

class T5Summarizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.model
        self.full_config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.name,
            cache_dir=self.full_config.general.model_cache_dir
        )
        
        # 2. 모델 초기화
        model_kwargs = {
            "cache_dir": self.full_config.general.model_cache_dir,
        }
        
        # torch_dtype 설정이 있는 경우 추가
        if hasattr(self.config, "model") and hasattr(self.config.model, "torch_dtype"):
            if self.config.model.torch_dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
        
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.config.name,
            **model_kwargs
        ).to(self.device)
        
        # 3. 모델 설정 업데이트
        self.model.config.update({
            "decoder_start_token_id": self.tokenizer.pad_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "bos_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "is_encoder_decoder": True,
            "task": "summarization",
            "max_length": self.config.generation.max_length,
            "min_length": self.config.generation.min_length,
            "length_penalty": self.config.generation.length_penalty,
            "early_stopping": self.config.generation.early_stopping
        })
        
        # 4. 특수 토큰 추가 (설정된 경우)
        if hasattr(self.config, 'tokenizer') and hasattr(self.config.tokenizer, 'special_tokens'):
            special_tokens_dict = {
                'additional_special_tokens': [
                    "#Person1#", "#Person2#", "#Person3#",
                    "#PhoneNumber#", "#Address#", "#PassportNumber#"
                ]
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))

    def batch_summarize(self, dialogues: List[str], sample_dialogue: str = None, 
                       sample_summary: str = None, prompt_version: str = "v1") -> List[str]:
        """배치 단위로 요약 생성"""
        summaries = []
        for dialogue in tqdm(dialogues, desc="Generating summaries"):
            # 프롬프트 생성
            prompt = self._build_prompt(dialogue, sample_dialogue, sample_summary, prompt_version)
            
            # 입력 인코딩
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.config.tokenizer.encoder_max_len
            ).to(self.device)
            
            # 요약 생성
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.generation.max_length,
                min_length=self.config.generation.min_length,
                num_beams=self.config.generation.num_beams,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                repetition_penalty=self.config.generation.repetition_penalty,
                length_penalty=self.config.generation.length_penalty,
                early_stopping=self.config.generation.early_stopping
            )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summaries.append(summary)
            
        return summaries

    def _build_prompt(self, dialogue: str, sample_dialogue: str = None, 
                     sample_summary: str = None, prompt_version: str = "v1") -> str:
        """프롬프트 생성"""
        # T5는 task prefix를 사용
        prefix = "summarize: "
        
        if sample_dialogue and sample_summary:  # few-shot
            return f"{prefix}Example Dialogue: {sample_dialogue}\nSummary: {sample_summary}\n\nDialogue: {dialogue}"
        else:  # zero-shot
            return f"{prefix}{dialogue}" 