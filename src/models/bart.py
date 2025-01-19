from transformers import AutoTokenizer, BartForConditionalGeneration
import torch.nn as nn
from dataclasses import dataclass
from transformers.utils import logging
from torch.quantization import quantize_dynamic
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

class BartSummarizer(nn.Module):
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
        
        # 2. 모델 초기화 - 불필요한 분류 관련 파라미터 제거
        model_kwargs = {
            "cache_dir": self.full_config.general.model_cache_dir,
            "problem_type": "seq2seq_lm",  # 명시적으로 seq2seq 태스크임을 지정
        }
        
        # torch_dtype 설정이 있는 경우 추가
        if hasattr(self.config, "model") and hasattr(self.config.model, "torch_dtype"):
            if self.config.model.torch_dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
        
        self.model = BartForConditionalGeneration.from_pretrained(
            self.config.name,
            **model_kwargs
        ).to(self.device)
        
        # 3. 특수 토큰 설정
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.forced_bos_token_id = self.tokenizer.bos_token_id
        self.model.config.forced_eos_token_id = self.tokenizer.eos_token_id
        
        # 4. 분류 관련 속성 제거
        if hasattr(self.model.config, "num_labels"):
            delattr(self.model.config, "num_labels")
        if hasattr(self.model.config, "id2label"):
            delattr(self.model.config, "id2label")
        if hasattr(self.model.config, "label2id"):
            delattr(self.model.config, "label2id")
        
        # 디버깅을 위한 설정 출력
        print("\nSpecial token settings:")
        print(f"decoder_start_token_id: {self.model.config.decoder_start_token_id}")
        print(f"bos_token_id: {self.model.config.bos_token_id}")
        print(f"eos_token_id: {self.model.config.eos_token_id}")
        
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

    def batch_summarize(self, dialogues: List[str], sample_dialogue: str = None, 
                       sample_summary: str = None, prompt_version: str = "v1") -> List[str]:
        """배치 단위로 요약 생성"""
        summaries = []
        for dialogue in tqdm(dialogues, desc="Generating summaries"):
            # 프롬프트 생성
            prompt = self._build_prompt(dialogue, sample_dialogue, sample_summary, prompt_version)
            
            # 입력 인코딩
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                  max_length=self.config.tokenizer.encoder_max_len).to(self.device)
            
            # 요약 생성
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.generation.max_new_tokens,
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
        if sample_dialogue and sample_summary:
            # few-shot 프롬프트
            return f"Dialogue:\n{sample_dialogue}\nSummary:\n{sample_summary}\n\nDialogue:\n{dialogue}\nSummary:"
        else:
            # zero-shot 프롬프트
            return f"Dialogue:\n{dialogue}\nSummary:" 