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
        
        # 모델과 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name,
            cache_dir=cfg.huggingface.cache_dir,
            token=cfg.huggingface.token
        )
        
        # 모델 초기화 - model 속성으로 저장
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            cache_dir=cfg.huggingface.cache_dir,
            token=cfg.huggingface.token,
            device_map="auto"
        )
        
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