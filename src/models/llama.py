from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.base_model import BaseModel
import torch
from typing import List, Optional
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
    prepare_model_for_kbit_training
)
from pathlib import Path
from omegaconf import OmegaConf
from transformers.generation import StoppingCriteriaList, StoppingCriteria
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

class StopWordsCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, prompt_length):
        self.stop_words = stop_words
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length  # 프롬프트 길이 저장
        
    def __call__(self, input_ids, scores, **kwargs):
        # 새로 생성된 부분만 디코딩
        new_text = self.tokenizer.decode(
            input_ids[0][self.prompt_length:], 
            skip_special_tokens=True
        )
        # 정확한 매칭만 체크 (endswith 제거)
        return any(word == new_text.strip() for word in self.stop_words)

class LlamaModel:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name,
            cache_dir=cfg.huggingface.cache_dir,
            token=cfg.huggingface.token,
            padding_side="left",  # LLaMA는 left padding 사용
            truncation_side="left"  # 왼쪽에서 자르기
        )
        
        # pad_token이 없으면 추가
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 양자화 설정
        if cfg.model.quantization.enabled:
            compute_dtype = getattr(torch, cfg.model.quantization.compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=cfg.model.quantization.precision == "int4",
                load_in_8bit=cfg.model.quantization.precision == "int8",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=cfg.model.quantization.double_quant,
                bnb_4bit_quant_type=cfg.model.quantization.quant_type
            )
        else:
            bnb_config = None
        
        # 모델 초기화
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            cache_dir=cfg.huggingface.cache_dir,
            token=cfg.huggingface.token,
            quantization_config=bnb_config,
            torch_dtype=getattr(torch, cfg.model.torch_dtype),
            trust_remote_code=cfg.model.trust_remote_code,
            device_map=cfg.model.device_map
        )
        
        # 양자화된 모델을 위한 준비
        if cfg.model.quantization.enabled:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=cfg.train.training.gradient_checkpointing
            )
        
        # LoRA 설정이 있으면 적용
        if cfg.model.lora.enabled:
            # 쉼표로 구분된 문자열을 리스트로 변환
            target_modules = cfg.model.lora.target_modules.split(',')
            
            lora_config = LoraConfig(
                r=cfg.model.lora.r,
                lora_alpha=cfg.model.lora.alpha,
                target_modules=target_modules,
                lora_dropout=cfg.model.lora.dropout,
                bias=cfg.model.lora.bias,
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        else:
            # LoRA를 사용하지 않는 경우 모든 파라미터를 학습 가능하도록 설정
            for param in self.model.parameters():
                param.requires_grad = True
        
        # Gradient Checkpointing 활성화 (메모리 절약)
        if cfg.train.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # 모델을 학습 모드로 설정
        self.model.train()
        
        # 프롬프트 템플릿 설정
        self.prompt_templates = cfg.prompt_templates
        
    def batch_summarize(self, dialogues, sample_dialogue=None, sample_summary=None, prompt_version="1"):
        """배치 단위로 요약 생성"""
        summaries = []
        template = self.prompt_templates[prompt_version]
        
        for dialogue in dialogues:
            if sample_dialogue and sample_summary:  # few-shot
                prompt = template["few_shot"].format(
                    sample_dialogue=sample_dialogue,
                    sample_summary=sample_summary,
                    dialogue=dialogue
                )
            else:  # zero-shot
                prompt = template["zero_shot"].format(dialogue=dialogue)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.model.generation.max_new_tokens,
                min_new_tokens=self.cfg.model.generation.min_new_tokens,
                num_beams=self.cfg.model.generation.num_beams,
                temperature=self.cfg.model.generation.temperature,
                top_p=self.cfg.model.generation.top_p,
                do_sample=self.cfg.model.generation.do_sample,
                length_penalty=self.cfg.model.generation.length_penalty,
                repetition_penalty=self.cfg.model.generation.repetition_penalty,
                no_repeat_ngram_size=self.cfg.model.generation.no_repeat_ngram_size,
                early_stopping=self.cfg.model.generation.early_stopping
            )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summaries.append(summary)
            
        return summaries

    def forward(self, **inputs):
        return self.model(**inputs)
        
    def generate(self, prompt: str) -> str:
        generation_config = OmegaConf.to_container(self.cfg.model.generation)
        
        # stop_sequences나 stop_tokens가 있으면 추가
        if 'stop_sequences' in generation_config:
            generation_config['stop'] = generation_config.pop('stop_sequences')
        elif 'stop_tokens' in generation_config:
            generation_config['stop'] = generation_config.pop('stop_tokens')
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            **generation_config,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id
        )
        
        # 생성된 텍스트에서 Summary: 이후 부분만 추출
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 첫 번째 문장만 추출
        if "." in generated_text:
            generated_text = generated_text.split(".")[0] + "."
        
        # assistant 등 불필요한 텍스트 제거
        unwanted_suffixes = ["assistant", "Assistant", "Summary:", "요약:"]
        for suffix in unwanted_suffixes:
            if generated_text.endswith(suffix):
                generated_text = generated_text[:-len(suffix)].strip()
        
        return generated_text.strip()
        
    def _build_prompt(self, dialogue: str, sample_dialogue: Optional[str] = None, 
                     sample_summary: Optional[str] = None, prompt_version: str = "v1") -> str:
        """프롬프트 생성"""
        if self.cfg.model.mode == "prompt":
            # prompt_version에서 'v' 접두사 제거하고 문자열로 유지 (v1 -> "1", v2 -> "2")
            version = str(prompt_version.replace('v', ''))
            templates = self.prompt_templates[version]
            
            if sample_dialogue and sample_summary:
                # few-shot 프롬프트
                return templates.few_shot.format(
                    dialogue=dialogue,
                    sample_dialogue=sample_dialogue,
                    sample_summary=sample_summary
                )
            else:
                # zero-shot 프롬프트
                return templates.zero_shot.format(dialogue=dialogue)
                
        else:  # finetune 모드
            return dialogue
        
    def prepare_inputs_for_generation(self, dialogue: str, summary: str = None):
        """학습/추론을 위한 입력 준비"""
        if summary:
            # 학습용 프롬프트
            prompt = f"Dialogue: {dialogue}\nSummary: {summary}"
        else:
            # 추론용 프롬프트
            prompt = f"Dialogue: {dialogue}\nSummary:"
            
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cfg.model.tokenizer.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        if summary:
            labels = inputs["input_ids"].clone()
            # dialogue 부분은 -100으로 마스킹
            summary_start = prompt.find("Summary: ") + len("Summary: ")
            summary_tokens = self.tokenizer(summary).input_ids
            labels[:, :-len(summary_tokens)] = -100
            inputs["labels"] = labels
            
        return inputs 
        
    def _clean_generated_text(self, text: str, prompt: str) -> str:
        """생성된 텍스트 정리"""
        # prompt 부분 제거
        if prompt in text:
            text = text[len(prompt):].strip()
        
        # 첫 번째 문장만 유지
        if "." in text:
            text = text.split(".")[0] + "."
        
        # 불필요한 접미사 제거
        unwanted_suffixes = ["assistant", "Assistant", "Summary:", "요약:", "\n"]
        for suffix in unwanted_suffixes:
            if text.endswith(suffix):
                text = text[:-len(suffix)].strip()
            
        return text.strip() 

    def _prepare_llama_dataset(self, df):
        """LLaMA 모델용 데이터셋 준비"""
        # 데이터 유효성 검사
        if 'dialogue' not in df.columns or 'summary' not in df.columns:
            raise ValueError("데이터셋에 'dialogue'와 'summary' 컬럼이 필요합니다.")
        
        # LLaMA 형식의 프롬프트 생성
        prompts = [
            f"### Dialogue:\n{dialogue}\n\n### Summary:\n{summary}"
            for dialogue, summary in zip(df['dialogue'], df['summary'])
        ]
        
        # 토크나이징
        tokenized = self.tokenizer(
            prompts,
            padding=self.config.model.tokenizer.padding,
            truncation=self.config.model.tokenizer.truncation,
            max_length=self.config.model.tokenizer.max_length,
            return_tensors=self.config.model.tokenizer.return_tensors
        )
        
        # 레이블 생성
        labels = tokenized['input_ids'].clone()
        
        # dialogue 부분을 -100으로 마스킹
        for i, prompt in enumerate(prompts):
            summary_start = prompt.find("### Summary:\n") + len("### Summary:\n")
            summary_tokens = self.tokenizer(prompt[summary_start:]).input_ids
            if len(summary_tokens) > 1:  # bos/eos 토큰 제외
                labels[i, :-len(summary_tokens)+1] = -100  # 마지막 eos 토큰 유지
        
        # 데이터셋 반환
        return DialogueDataset(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
            labels=labels
        ) 