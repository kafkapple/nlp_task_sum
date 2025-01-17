from typing import List, Dict, Any
from openai import OpenAI
import time
from tqdm import tqdm

class SolarAPI:
    def __init__(self, config: Dict[str, Any]):
        self.client = OpenAI(
            api_key=config.model.api.api_key,
            base_url=config.model.api.base_url
        )
        self.config = config
        self.model_name = config.model.name
        self.temperature = config.model.api.temperature
        self.top_p = config.model.api.top_p
    
    def build_prompt(self, dialogue: str, sample_dialogue: str = None, 
                    sample_summary: str = None, prompt_version: str = "v1"):
        """Build prompt based on version"""
        templates = self.config.model.prompt_templates[prompt_version]
        
        if prompt_version == "v1":
            if sample_dialogue and sample_summary:
                user_prompt = templates.few_shot.format(
                    instruction=templates.instruction,
                    dialogue=dialogue,
                    sample_dialogue=sample_dialogue,
                    sample_summary=sample_summary
                )
            else:
                user_prompt = templates.zero_shot.format(dialogue=dialogue)
                
            return [
                {"role": "system", "content": templates.system},
                {"role": "user", "content": user_prompt}
            ]
        else:  # v2
            messages = [{"role": "system", "content": templates.system}]
            
            if sample_dialogue and sample_summary:
                messages.extend([
                    {
                        "role": "user",
                        "content": templates.few_shot.user.format(
                            instruction=templates.instruction,
                            sample_dialogue=sample_dialogue
                        )
                    },
                    {
                        "role": "assistant",
                        "content": templates.few_shot.assistant.format(
                            sample_summary=sample_summary
                        )
                    },
                    {
                        "role": "user",
                        "content": templates.few_shot.final_user.format(
                            dialogue=dialogue
                        )
                    }
                ])
            else:
                messages.append({
                    "role": "user",
                    "content": templates.zero_shot.format(dialogue=dialogue)
                })
            
            return messages
    
    def summarize(self, dialogue: str, sample_dialogue: str = None, 
                 sample_summary: str = None, prompt_version: str = "v1") -> str:
        """Generate summary using Solar API"""
        messages = self.build_prompt(
            dialogue, 
            sample_dialogue, 
            sample_summary, 
            prompt_version
        )
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p
        )
        
        return response.choices[0].message.content
        
    def batch_summarize(self, dialogues: List[str], sample_dialogue: str = None, 
                       sample_summary: str = None, batch_size: int = 100, prompt_version: str = "v1") -> List[str]:
        """Generate summaries for a batch of dialogues"""
        summaries = []
        start_time = time.time()
        
        for i, dialogue in enumerate(tqdm(dialogues)):
            summaries.append(self.summarize(dialogue, sample_dialogue, sample_summary, prompt_version))
            
            if (i + 1) % batch_size == 0:
                elapsed_time = time.time() - start_time
                if elapsed_time < 60:
                    wait_time = 60 - elapsed_time + 5
                    time.sleep(wait_time)
                start_time = time.time()
                
        return summaries 