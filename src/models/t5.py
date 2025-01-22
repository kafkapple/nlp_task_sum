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
        
        # 1. Quantization 설정
        quantization_config = ModelFactory._create_quantization_config(config)
        
        # 2. 모델 초기화
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # 3. LoRA 또는 Partial Training 적용
        if config.model.mode == "finetune":
            if config.model.get("partial_training", False):
                self._freeze_layers(
                    num_layers_to_train=config.model.get("num_layers_to_train", 2)
                )
            elif config.model.get("lora", False):
                peft_config = ModelFactory._create_peft_config(config.model.family)
                self.model = get_peft_model(self.model, peft_config)
        
        # 4. Gradient Checkpointing 활성화
        if config.train.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
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