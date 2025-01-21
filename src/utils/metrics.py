from rouge import Rouge
from typing import Dict, List
import wandb
import numpy as np

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

class TrainerMetrics:
    def __init__(self, tokenizer, remove_tokens):
        self.tokenizer = tokenizer
        self.remove_tokens = remove_tokens
        self.rouge = Rouge()
        
    def __call__(self, pred):
        try:
            labels_ids = pred.label_ids
            pred_ids = pred.predictions
            
            if isinstance(pred_ids, tuple):
                pred_ids = pred_ids[0]
            
            print("\n=== Metric Calculation Debug ===")
            print(f"Predictions shape: {pred_ids.shape}")
            print(f"Labels shape: {labels_ids.shape}")
            
            # -100을 pad_token_id로 변환
            labels_ids = labels_ids.copy()  # 원본 데이터 보존
            labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
            
            # 음수 값 처리
            pred_ids = np.clip(pred_ids, a_min=0, a_max=None)
            
            # 디코딩
            pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(
                labels_ids, 
                skip_special_tokens=True
            )
            
            # 샘플 출력
            print("\n=== Sample Predictions ===")
            for i in range(min(3, len(pred_str))):
                print(f"\nSample {i+1}:")
                print(f"Prediction: {pred_str[i]}")
                print(f"Gold Summary: {label_str[i]}")
            
            # 특수 토큰 제거
            for token in self.remove_tokens:
                pred_str = [p.replace(token, '').strip() for p in pred_str]
                label_str = [l.replace(token, '').strip() for l in label_str]
            
            # ROUGE 점수 계산
            all_scores = []
            for p, l in zip(pred_str, label_str):
                if not p.strip():
                    print(f"\nEmpty prediction found: '{p}'")
                    p = "empty"
                if not l.strip():
                    print(f"\nEmpty label found: '{l}'")
                    l = "empty"
                    
                try:
                    score = self.rouge.get_scores(p, l)[0]
                    all_scores.append(score)
                except Exception as e:
                    print(f"\nError calculating ROUGE for:")
                    print(f"Prediction: '{p}'")
                    print(f"Label: '{l}'")
                    print(f"Error: {str(e)}")
                    all_scores.append({'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}})
            
            # 평균 계산
            avg_scores = {
                'eval_rouge1_f1': np.mean([s['rouge-1']['f'] for s in all_scores]),
                'eval_rouge2_f1': np.mean([s['rouge-2']['f'] for s in all_scores]),
                'eval_rougeL_f1': np.mean([s['rouge-l']['f'] for s in all_scores])
            }
            
            print("\n=== Final Scores ===")
            print(avg_scores)
            
            return avg_scores
            
        except Exception as e:
            print(f"\n=== Error in Metric Calculation ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'eval_rouge1_f1': 0.0,
                'eval_rouge2_f1': 0.0,
                'eval_rougeL_f1': 0.0
            } 