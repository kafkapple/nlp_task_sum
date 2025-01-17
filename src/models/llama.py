from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.base_model import BaseModel
import torch
from typing import List, Optional
from peft import get_peft_model, LoraConfig, TaskType
from pathlib import Path
from omegaconf import OmegaConf

class LlamaModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config.model  # 모델 관련 설정
        self.full_config = config   # 전체 설정 (prompt_templates 등 접근용)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_model()
        
        # finetune 모드일 때 LoRA 설정
        if self.config.mode == "finetune":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora.r,  # config.model의 lora 설정 사용
                lora_alpha=self.config.lora.alpha,
                lora_dropout=self.config.lora.dropout,
                target_modules=self.config.lora.target_modules
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
    
    def _init_model(self):
        # cache_dir 설정 및 생성
        cache_dir = Path(self.full_config.general.model_cache_dir)  # full_config에서 경로 가져오기
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"모델 캐시 디렉토리: {cache_dir}")
        
        # 토크나이저 초기화
        try:
            print(f"토크나이저 로드 중... ({self.config.name})")  # config.model.name -> config.name
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.name,  # config.model.name -> config.name
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=False
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("토크나이저 로드 완료")
            
            # 모델 초기화
            print("모델 로드 중...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.name,  # config.model.name -> config.name
                trust_remote_code=True,
                torch_dtype=getattr(torch, "float16"),
                device_map="auto",
                cache_dir=cache_dir,
                local_files_only=False
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
        
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generation_config = OmegaConf.to_container(self.config.generation)
        
        # 특수 토큰 ID 추가
        generation_config.update({
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'bos_token_id': self.tokenizer.bos_token_id
        })
        
        outputs = self.model.generate(
            **inputs,
            **generation_config
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
        if self.config.mode == "prompt":
            # prompt_version에서 'v' 접두사 제거하고 문자열로 유지 (v1 -> "1", v2 -> "2")
            version = str(prompt_version.replace('v', ''))
            templates = self.full_config.prompt_templates[version]
            
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
        
    def batch_summarize(self, dialogues: List[str], sample_dialogue: str = None, 
                       sample_summary: str = None, **kwargs) -> List[str]:
        summaries = []
        
        # yaml 설정을 직접 사용
        generation_config = self.config.generation
        
        for dialogue in dialogues:
            prompt = self._build_prompt(dialogue, sample_dialogue, sample_summary)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **OmegaConf.to_container(generation_config),  # yaml 설정을 모두 적용
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # prompt 부분과 불필요한 접미사 제거
            summary = self._clean_generated_text(summary, prompt)
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