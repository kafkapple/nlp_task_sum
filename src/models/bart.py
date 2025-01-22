from transformers import AutoTokenizer, BartForConditionalGeneration
import torch.nn as nn
from dataclasses import dataclass
from transformers.utils import logging
from torch.quantization import quantize_dynamic
import torch
from typing import List
from tqdm import tqdm
from peft import get_peft_model
from src.models.model_factory import ModelFactory
from transformers import AutoModelForSeq2SeqLM
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
        
        # 2. 모델 초기화 - seq2seq 설정
        model_kwargs = {
            "cache_dir": self.full_config.general.model_cache_dir,
        }
        
        # torch_dtype 설정이 있는 경우 추가
        if hasattr(self.config, "model") and hasattr(self.config.model, "torch_dtype"):
            if self.config.model.torch_dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
        
        # self.model = BartForConditionalGeneration.from_pretrained(
        #     self.config.name,
        #     **model_kwargs
        # ).to(self.device)
        # 1. Quantization 설정
        quantization_config = ModelFactory._create_quantization_config(config)
        
        # 2. 모델 초기화
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
            # 3. LoRA 적용
        if config.model.mode == "finetune":
            if config.model.get("partial_training", False):  # 설정에서 partial training 활성화된 경우
                self._freeze_layers(
                    num_layers_to_train=config.model.get("num_layers_to_train", 2)
                )
            elif config.model.get("lora", False):  # LoRA 사용하는 경우
                peft_config = ModelFactory._create_peft_config(config.model.family)
                self.model = get_peft_model(self.model, peft_config)
        
        # 4. Gradient Checkpointing 활성화
        if config.train.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        # 3. 모델 설정 업데이트
        self.model.config.update({
            "decoder_start_token_id": self.tokenizer.bos_token_id,
            "forced_bos_token_id": self.tokenizer.bos_token_id,
            "forced_eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "is_encoder_decoder": True,
            "task": "summarization",
            "problem_type": "seq2seq_lm",
            # 분류 관련 설정 제거 - None 대신 기본값 사용
            "num_labels": 2,  # 기본값으로 설정
            "id2label": {0: "LABEL_0", 1: "LABEL_1"},
            "label2id": {"LABEL_0": 0, "LABEL_1": 1}
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

    def _freeze_layers(self, num_layers_to_train: int = 2):
        """특정 레이어만 학습하도록 설정"""
        # 모든 파라미터 freeze
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Encoder의 마지막 N층 unfreeze
        encoder_layers = self.model.model.encoder.layers
        for i in range(-num_layers_to_train, 0):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True
        
        # Decoder의 마지막 N층 unfreeze
        decoder_layers = self.model.model.decoder.layers
        for i in range(-num_layers_to_train, 0):
            for param in decoder_layers[i].parameters():
                param.requires_grad = True
            # Cross-attention도 학습
            for param in decoder_layers[i].encoder_attn.parameters():
                param.requires_grad = True 