from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch.nn as nn
from dataclasses import dataclass
from transformers.utils import logging
import torch
from typing import List
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_model
from src.models.model_factory import ModelFactory
from src.models.base_model import BaseModel

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

class T5Summarizer(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config.model
        self.full_config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Quantization 설정
        quantization_config = ModelFactory._create_quantization_config(config)
        
        # 2. 모델 초기화
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # 1. 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.name,
            cache_dir=self.full_config.general.model_cache_dir
        )
        
        # 2. 모델 설정 업데이트
        self.model.config.update({
            "decoder_start_token_id": self.tokenizer.pad_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "bos_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "is_encoder_decoder": True,
            "task": "summarization",
            "max_new_tokens": self.config.generation.max_new_tokens,
            "min_new_tokens": self.config.generation.min_new_tokens,
            "length_penalty": self.config.generation.length_penalty,
            "early_stopping": self.config.generation.early_stopping
        })
        
        # 3. 특수 토큰 추가 - BaseModel의 메서드 사용
        if hasattr(self.config.tokenizer, 'special_tokens'):
            self._add_special_tokens(
                self.tokenizer,
                self.config.tokenizer.special_tokens.additional_special_tokens
            )
        
        # 4. LoRA 또는 Partial Training 적용
        if config.model.mode == "finetune":
            if config.model.get("partial_training", False):
                self._freeze_layers(
                    num_layers_to_train=config.model.get("num_layers_to_train", 2)
                )
            elif config.model.get("lora", False):
                peft_config = ModelFactory._create_peft_config(config.model.family)
                self.model = get_peft_model(self.model, peft_config)
        
        # 5. Gradient Checkpointing 활성화
        if config.train.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, **inputs):
        """Forward pass for training"""
        return self.model(**inputs)

    def generate(self, **inputs):
        """Generation for inference"""
        return self.model.generate(**inputs)

    def batch_summarize(self, dialogues: List[str], sample_dialogue: str = None, 
                       sample_summary: str = None, prompt_version: str = None) -> List[str]:
        """배치 단위 요약 생성"""
        summaries = []
        
        for idx, dialogue in enumerate(tqdm(dialogues, desc="Generating summaries")):
            # 프롬프트 생성
            prompt = self._build_prompt(
                dialogue=dialogue, 
                sample_dialogue=sample_dialogue, 
                sample_summary=sample_summary, 
                prompt_version=self.cfg.prompt.version
            )
            
            # 첫 번째 대화의 프롬프트 출력
            if idx == 0:
                print("\n=== Sample Prompt ===")
                print("-" * 50)
                print(prompt)
                print("-" * 50)
                print("\n=== Token Length ===")
                print(f"Input tokens: {len(self.tokenizer.encode(dialogue))}")
                if sample_dialogue:
                    print(f"Sample tokens: {len(self.tokenizer.encode(sample_dialogue))}")
                print("==================\n")
            
            # 입력 텍스트 길이 계산 (토큰 기준)
            input_length = len(self.tokenizer.encode(dialogue))
            
            # 목표 길이 계산 (ratio 기반)
            target_length = int(input_length * self.config.generation.length_ratio)
            
            # max_new_tokens와 min_new_tokens를 동적으로 설정
            max_tokens = min(int(target_length * 1.5), self.config.tokenizer.decoder_max_len)
            min_tokens = max(int(target_length * 0.5), 10)  # 최소 10 토큰
            
            # 토크나이징
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.config.tokenizer.encoder_max_len
            ).to(self.device)
            
            # 요약 생성
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                num_beams=self.config.generation.num_beams,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                repetition_penalty=self.config.generation.repetition_penalty,
                length_penalty=self.config.generation.length_penalty,
                early_stopping=self.config.generation.early_stopping
            )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summaries.append(summary)
            
            # 첫 번째 대화의 요약 결과 출력
            if idx == 0:
                print("\n=== First Summary ===")
                print(f"Generated Summary: {summary}")
                print(f"Summary tokens: {len(self.tokenizer.encode(summary))}")
                print("====================\n")
            
        return summaries

    def _build_prompt(self, dialogue: str, sample_dialogue: str = None, 
                      sample_summary: str = None, prompt_version: str = "v1") -> str:
        """Build prompt for the model based on the dialogue and sample data."""
        print("\n=== Prompt Template Information ===")
        print(f"Version: {prompt_version}")
        print(f"Mode: {'Few-shot' if sample_dialogue else 'Zero-shot'}")
        print(f"Few-shot enabled: {sample_dialogue and sample_summary}")
        
        if prompt_version == "v1":
            # Version 1: Original prompt template
            if sample_dialogue and sample_summary:
                print("\nFew-shot prompt template v1:")
                prompt = f"다음은 대화문과 요약문의 예시입니다:\n대화문:\n{sample_dialogue}\n요약문:\n{sample_summary}\n\n이제 다음 대화문을 요약해주세요:\n대화문:\n{dialogue}"
                print(f"\nSample dialogue:\n{sample_dialogue}")
                print(f"\nSample summary:\n{sample_summary}")
            else:
                print("\nZero-shot prompt template v1:")
                prompt = f"다음 대화문을 요약해주세요:\n대화문:\n{dialogue}"
        else:
            # Version 2: Enhanced prompt template
            if sample_dialogue and sample_summary:
                print("\nFew-shot prompt template v2:")
                prompt = f"""다음은 대화문과 요약문의 예시입니다.
예시를 잘 읽고 아래 대화문을 요약해주세요.

[예시 대화문]
{sample_dialogue}

[예시 요약문]
{sample_summary}

이제 다음 대화문을 위 예시처럼 요약해주세요.
100자 이내로 작성하고, 구어체는 문어체로 바꿔주세요.

[대화문]
{dialogue}

[요약문]"""
                print(f"\nSample dialogue:\n{sample_dialogue}")
                print(f"\nSample summary:\n{sample_summary}")
            else:
                print("\nZero-shot prompt template v2:")
                prompt = f"""다음 대화문을 요약해주세요.
100자 이내로 작성하고, 구어체는 문어체로 바꿔주세요.

[대화문]
{dialogue}

[요약문]"""

        print(f"\nInput dialogue:\n{dialogue}")
        print(f"\nFinal prompt:\n{prompt}")
        print("===========================\n")
        return prompt

    def _freeze_layers(self, num_layers_to_train: int = 2):
        """특정 레이어만 학습하도록 설정"""
        # 모든 파라미터 freeze
        for param in self.model.parameters():
            if param.dtype in [torch.float16, torch.float32, torch.float64]:
                param.requires_grad = False
        
        # Encoder의 마지막 N층 unfreeze
        encoder_layers = self.model.encoder.block
        for i in range(-num_layers_to_train, 0):
            for param in encoder_layers[i].parameters():
                if param.dtype in [torch.float16, torch.float32, torch.float64]:
                    param.requires_grad = True
        
        # Decoder의 마지막 N층 unfreeze
        decoder_layers = self.model.decoder.block
        for i in range(-num_layers_to_train, 0):
            for param in decoder_layers[i].parameters():
                if param.dtype in [torch.float16, torch.float32, torch.float64]:
                    param.requires_grad = True
            # Cross-attention도 학습
            for param in decoder_layers[i].layer[1].parameters():  # T5의 cross-attention은 layer[1]에 있음
                if param.dtype in [torch.float16, torch.float32, torch.float64]:
                    param.requires_grad = True
                    
        # Shared embedding layer는 항상 학습되도록 설정
        for param in self.model.shared.parameters():
            if param.dtype in [torch.float16, torch.float32, torch.float64]:
                param.requires_grad = True
                
        # Final layer norm layers도 학습되도록 설정
        if hasattr(self.model.encoder, 'final_layer_norm'):
            for param in self.model.encoder.final_layer_norm.parameters():
                if param.dtype in [torch.float16, torch.float32, torch.float64]:
                    param.requires_grad = True
        if hasattr(self.model.decoder, 'final_layer_norm'):
            for param in self.model.decoder.final_layer_norm.parameters():
                if param.dtype in [torch.float16, torch.float32, torch.float64]:
                    param.requires_grad = True 