import pandas as pd
import numpy as np
from typing import Dict, List, Union
import random

class FewShotSelector:
    """Few-shot 학습을 위한 예제 선택기"""
    
    @staticmethod
    def select_examples(train_df: pd.DataFrame, cfg) -> Dict[str, Union[str, List[str]]]:
        """학습 데이터에서 few-shot 예제 선택
        
        Args:
            train_df: 학습 데이터 DataFrame
            cfg: 설정 객체
            
        Returns:
            Dict: {
                "dialogue": 선택된 대화,
                "summary": 선택된 요약
            }
        """
        n_samples = cfg.custom_config.n_few_shot_samples
        selection_method = cfg.custom_config.get("few_shot_selection", "random")
        
        if selection_method == "random":
            # 랜덤 선택
            selected = train_df.sample(n=n_samples, random_state=cfg.general.seed)
        elif selection_method == "length":
            # 길이 기반 선택 (중간 길이의 예제 선택)
            train_df['dialogue_len'] = train_df['dialogue'].str.len()
            median_len = train_df['dialogue_len'].median()
            selected = train_df.iloc[(train_df['dialogue_len'] - median_len).abs().argsort()[:n_samples]]
        elif selection_method == "baseline":
            # 첫 번째 예제 사용
            selected = train_df.head(n_samples)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
            
        if n_samples == 1:
            return {
                "dialogue": selected.iloc[0]['dialogue'],
                "summary": selected.iloc[0]['summary']
            }
        else:
            return {
                "dialogue": selected['dialogue'].tolist(),
                "summary": selected['summary'].tolist()
            }
            
    @staticmethod
    def print_example(dialogue: str, summary: str):
        """선택된 예제 출력"""
        print("\n=== Selected Few-shot Example ===")
        print("Dialogue:")
        print(dialogue)
        print("\nSummary:")
        print(summary)
        print("===============================\n")

    @staticmethod
    def _select_by_similarity(df, n_samples):
        """유사도 기반 선택 로직"""
        pass
        
    @staticmethod
    def _select_diverse_samples(df, n_samples):
        """다양성 기반 선택 로직"""
        pass 