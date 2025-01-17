from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.base_model import BaseModel
import torch
from typing import List, Optional
from peft import get_peft_model, LoraConfig, TaskType
from pathlib import Path

class LlamaModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_model()
        
        # finetune 모드일 때 LoRA 설정
        if self.config.model.mode == "finetune":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora.r,
                lora_alpha=self.config.lora.alpha,
                lora_dropout=self.config.lora.dropout,
                target_modules=self.config.lora.target_modules
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
    
    def _init_model(self):
        # cache_dir 설정 및 생성
        cache_dir = Path(self.config.general.model_cache_dir)  # general에서 경로 가져오기
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"모델 캐시 디렉토리: {cache_dir}")
        
        # 토크나이저 초기화
        try:
            print(f"토크나이저 로드 중... ({self.config.model.name})")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.name,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=False  # 필요한 경우 온라인에서 다운로드
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("토크나이저 로드 완료")
            
            # 모델 초기화
            print("모델 로드 중...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.name,
                trust_remote_code=True,
                torch_dtype=getattr(torch, "float16"),
                device_map="auto",
                cache_dir=cache_dir,
                local_files_only=False  # 필요한 경우 온라인에서 다운로드
            )
            print("모델 로드 완료")
            
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")
            print(f"캐시 디렉토리 내용:")
            if cache_dir.exists():
                for item in cache_dir.iterdir():
                    print(f" - {item}")
            raise
    
    def forward(self, **inputs):
        return self.model(**inputs)
        
    def generate(self, **inputs):
        return self.model.generate(**inputs)
        
    def _build_prompt(self, dialogue: str, sample_dialogue: Optional[str] = None, 
                     sample_summary: Optional[str] = None, prompt_version: str = "v1") -> str:
        if self.config.model.mode == "prompt":
            templates = self.config.prompt_templates[prompt_version]
            
            if sample_dialogue and sample_summary:
                # few-shot prompt
                return templates.few_shot.format(
                    dialogue=dialogue,
                    sample_dialogue=sample_dialogue,
                    sample_summary=sample_summary
                )
            else:
                # zero-shot prompt
                return templates.zero_shot.format(dialogue=dialogue)
                
        else:  # finetune 모드
            return dialogue
        
    def batch_summarize(self, dialogues: List[str], sample_dialogue: str = None, 
                       sample_summary: str = None, **kwargs) -> List[str]:
        summaries = []
        
        # generation 설정 가져오기 (없으면 기본값 사용)
        generation_config = getattr(self.config.model, 'generation', {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'repetition_penalty': 1.1
        })
        
        for dialogue in dialogues:
            prompt = self._build_prompt(dialogue, sample_dialogue, sample_summary)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_config.get('max_new_tokens', 512),
                    temperature=generation_config.get('temperature', 0.7),
                    top_p=generation_config.get('top_p', 0.9),
                    repetition_penalty=generation_config.get('repetition_penalty', 1.1),
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # prompt 부분 제거
            summary = summary[len(prompt):].strip()
            summaries.append(summary)
            
        return summaries 
        
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
            max_length=self.config.tokenizer.max_length,
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