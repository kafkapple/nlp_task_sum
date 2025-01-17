from openai import OpenAI
from typing import List, Dict, Any
import time
from tqdm import tqdm
import os
from dotenv import load_dotenv
from omegaconf import OmegaConf

class SolarAPI:
    def __init__(self, config):
        load_dotenv()
        api_key = os.getenv('UPSTAGE_API_KEY', config.model.api.api_key)
        print(f"API Key 확인: {api_key[:8]}...")  # API 키의 앞부분만 출력
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=config.model.api.base_url
        )
        self.config = config.model
        self.model_name = config.model.api.model  # api.model에서 모델명 가져오기
        # generation 설정을 직접 사용
        self.generation_config = OmegaConf.to_container(config.model.generation)
        
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
            print(f"API 요청 설정:")
            print(f"- temperature: {self.generation_config['temperature']}")
            print(f"- top_p: {self.generation_config['top_p']}")
            print(f"- max_tokens: {self.generation_config['max_new_tokens']}")
            print(f"- stop: {self.generation_config['stop_sequences']}")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.generation_config['temperature'],
                top_p=self.generation_config['top_p'],
                max_tokens=self.generation_config['max_new_tokens'],
                stop=self.generation_config['stop_sequences'],
                stream=False
            )
            
            summary = response.choices[0].message.content.strip()
            if not summary:
                return "요약 생성에 실패했습니다."
            return summary
            
        except Exception as e:
            print(f"\nAPI 호출 중 상세 오류:")
            print(f"- 오류 유형: {type(e).__name__}")
            print(f"- 오류 메시지: {str(e)}")
            print(f"- 모델명: {self.model_name}")
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