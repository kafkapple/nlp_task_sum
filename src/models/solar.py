from openai import OpenAI
from typing import List, Dict, Any
import time
from tqdm import tqdm
import os
from dotenv import load_dotenv
from omegaconf import OmegaConf

class SolarAPI:
    def __init__(self, config: Dict[str, Any]):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv('UPSTAGE_API_KEY', config.model.api.api_key),
            base_url=config.model.api.base_url
        )
        self.config = config.model
        self.model_name = "solar-1-mini-chat"
        self.temperature = config.model.api.temperature
        self.top_p = config.model.api.top_p
        
    def _build_messages(self, dialogue: str, sample_dialogue: str = None, 
                       sample_summary: str = None, prompt_version: str = "v1") -> List[Dict[str, str]]:
        """메시지 구성"""
        version = str(prompt_version.replace('v', ''))
        templates = self.config.prompt_templates[version]
        
        messages = [
            {
                "role": "system",
                "content": str(templates.system)
            }
        ]
        
        if sample_dialogue and sample_summary:
            if version == "2":
                # First turn: 예시 대화
                messages.append({
                    "role": "user",
                    "content": str(templates.few_shot.user).format(
                        instruction=str(templates.instruction),
                        sample_dialogue=sample_dialogue
                    )
                })
                # Second turn: 예시 요약
                messages.append({
                    "role": "assistant",
                    "content": str(templates.few_shot.assistant).format(
                        sample_summary=sample_summary
                    )
                })
                # Final turn: 실제 요약할 대화
                messages.append({
                    "role": "user",
                    "content": str(templates.few_shot.final_user).format(
                        dialogue=dialogue
                    )
                })
            else:
                # v1 형식: 단일 프롬프트에 모든 내용 포함
                messages.append({
                    "role": "user",
                    "content": str(templates.few_shot).format(
                        dialogue=dialogue,
                        sample_dialogue=sample_dialogue,
                        sample_summary=sample_summary
                    )
                })
        else:
            # zero-shot: 예시 없이 바로 요약
            messages.append({
                "role": "user",
                "content": str(templates.zero_shot).format(dialogue=dialogue)
            })
            
        return messages
        
    def summarize(self, dialogue: str, sample_dialogue: str = None, 
                 sample_summary: str = None, prompt_version: str = "v1") -> str:
        """Generate summary using Solar API"""
        messages = self._build_messages(
            dialogue, 
            sample_dialogue, 
            sample_summary, 
            prompt_version
        )
        
        try:
            # OmegaConf ListConfig를 일반 리스트로 변환
            generation_config = OmegaConf.to_container(self.config.api.generation)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=generation_config['max_tokens'],
                stop=list(generation_config['stop']),  # ListConfig를 list로 변환
                stream=False
            )
            
            summary = response.choices[0].message.content.strip()
            if not summary:
                return "요약 생성에 실패했습니다."
            return summary
            
        except Exception as e:
            print(f"API 호출 중 오류 발생: {str(e)}")
            return "API 오류가 발생했습니다."
        
    def batch_summarize(self, dialogues: List[str], sample_dialogue: str = None, 
                       sample_summary: str = None, prompt_version: str = "v1") -> List[str]:
        """Generate summaries for a batch of dialogues"""
        summaries = []
        
        for dialogue in tqdm(dialogues, desc="Generating summaries"):
            summary = self.summarize(
                dialogue, 
                sample_dialogue, 
                sample_summary, 
                prompt_version
            )
            summaries.append(summary)
            time.sleep(1)  # API 호출 간격 조절
            
        return summaries 