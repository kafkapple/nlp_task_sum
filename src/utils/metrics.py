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

def compute_metrics(tokenizer, pred, config):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    # 디코딩
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    # 특수 토큰 제거
    for token in config.inference.remove_tokens:
        pred_str = [pred.replace(token, '').strip() for pred in pred_str]
        label_str = [label.replace(token, '').strip() for label in label_str]
    
    # 빈 문자열 처리
    pred_str = [pred if pred else "empty" for pred in pred_str]
    label_str = [label if label else "empty" for label in label_str]
    
    try:
        rouge = Rouge()
        scores = rouge.get_scores(pred_str, label_str, avg=True)
        
        return {
            'rouge1_f1': scores['rouge-1']['f'],
            'rouge2_f1': scores['rouge-2']['f'],
            'rougeL_f1': scores['rouge-l']['f']
        }
    except ValueError as e:
        print(f"ROUGE 계산 중 오류 발생: {str(e)}")
        print(f"Sample predictions: {pred_str[:3]}")
        print(f"Sample labels: {label_str[:3]}")
        return {
            'rouge1_f1': 0.0,
            'rouge2_f1': 0.0,
            'rougeL_f1': 0.0
        } 