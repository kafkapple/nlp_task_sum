from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseModel(ABC):
    @abstractmethod
    def forward(self, **inputs):
        pass
        
    @abstractmethod
    def generate(self, **inputs):
        pass

    def _add_special_tokens(self, tokenizer, special_tokens):
        """특수 토큰 추가를 위한 공통 메서드"""
        if special_tokens:
            special_tokens_dict = {
                'additional_special_tokens': [str(token) for token in special_tokens]
            }
            num_added = tokenizer.add_special_tokens(special_tokens_dict)
            if num_added > 0:
                self.model.resize_token_embeddings(len(tokenizer))
            return num_added
        return 0 

    def _build_prompt(self, dialogue: str, sample_dialogue: str = None, 
                     sample_summary: str = None, prompt_version: str = "v1") -> str:
        """프롬프트 생성 - 여러 few-shot 예제 지원"""
        if sample_dialogue and sample_summary:
            # few-shot 예제 처리
            few_shot_examples = ""
            if isinstance(sample_dialogue, list) and isinstance(sample_summary, list):
                # 여러 예제 처리
                for d, s in zip(sample_dialogue, sample_summary):
                    few_shot_examples += f"대화:\n{d}\n요약:\n{s}\n\n"
            else:
                # 단일 예제 처리
                few_shot_examples = f"대화:\n{sample_dialogue}\n요약:\n{sample_summary}\n\n"

            # 최종 프롬프트 구성
            prompt = (
                f"{few_shot_examples}"
                f"이제 아래 대화를 위 예시들과 비슷한 스타일로 요약해주세요.\n"
                f"대화:\n{dialogue}\n"
                f"요약:"
            )
            return prompt
        else:
            # zero-shot 프롬프트
            return f"다음 대화를 요약해주세요.\n대화:\n{dialogue}\n요약:" 