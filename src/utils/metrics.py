from rouge import Rouge
from typing import Dict, List
import wandb
class Metrics:
    def __init__(self):
        self.rouge = Rouge()
        
    def compute_metrics(self, pred: str, gold: str) -> Dict[str, float]:
        """Compute ROUGE scores between prediction and ground truth"""
        results = self.rouge.get_scores(pred, gold, avg=True)
        return {key: value["f"] for key, value in results.items()}
        
    def compute_batch_metrics(self, preds: List[str], golds: List[str]) -> Dict[str, float]:
        """Compute average metrics for a batch of predictions"""
        scores = []
        raw_scores = []
        for pred, gold in zip(preds, golds):
            results = self.compute_metrics(pred, gold)
            avg_score = sum(results.values()) / len(results)
            scores.append(avg_score)
            raw_scores.append(results)
        avg_score_dict = {}
        for key in raw_scores[0].keys():
            avg_score_dict[key] = sum(score[key] for score in raw_scores) / len(raw_scores)
            
        return sum(scores) / len(scores), avg_score_dict 
from rouge import Rouge
import numpy as np

class TrainerMetrics:
    def __init__(self, tokenizer, remove_tokens):
        self.tokenizer = tokenizer
        self.remove_tokens = remove_tokens
        self.rouge = Rouge()
        
    def __call__(self, pred):
        """
        Trainer용 메트릭 계산 함수
        pred: EvalPrediction 객체 (predictions, label_ids 속성을 가짐)
        """
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        
        # 음수 값 처리
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]  # 첫 번째 요소만 사용
            
        # 음수 값을 pad_token_id로 변환
        pred_ids = pred_ids.clip(min=0)  # 음수를 0으로 변환
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        
        try:
            # 디코딩
            pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
            
            # 특수 토큰 제거
            for token in self.remove_tokens:
                pred_str = [pred.replace(token, '').strip() for pred in pred_str]
                label_str = [label.replace(token, '').strip() for label in label_str]
            
            # 빈 문자열 처리
            pred_str = [pred if pred else "empty" for pred in pred_str]
            label_str = [label if label else "empty" for label in label_str]
            
            scores = self.rouge.get_scores(pred_str, label_str, avg=True)
            
            return {
                'rouge1_f1': scores['rouge-1']['f'],
                'rouge2_f1': scores['rouge-2']['f'],
                'rougeL_f1': scores['rouge-l']['f']
            }
            
        except Exception as e:
            print(f"메트릭 계산 중 오류 발생: {str(e)}")
            print(f"예측 ID 형태: {type(pred_ids)}, 크기: {pred_ids.shape}")
            print(f"레이블 ID 형태: {type(labels_ids)}, 크기: {labels_ids.shape}")
            print(f"예측 ID 범위: [{pred_ids.min()}, {pred_ids.max()}]")
            print(f"레이블 ID 범위: [{labels_ids.min()}, {labels_ids.max()}]")
            return {
                'rouge1_f1': 0.0,
                'rouge2_f1': 0.0,
                'rougeL_f1': 0.0
            } 