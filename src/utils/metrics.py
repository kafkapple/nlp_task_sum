from rouge import Rouge
from typing import Dict, List
import wandb
import numpy as np
from transformers import PreTrainedTokenizerBase
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
import traceback

class Metrics:
    def __init__(self, config: DictConfig):
        self.config = config
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
    """학습 및 평가를 위한 메트릭 계산 클래스"""
    
    def __init__(self, config: DictConfig, tokenizer: PreTrainedTokenizerBase, remove_tokens: List[str] = None):
        self.config = config
        self.tokenizer = tokenizer
        self.remove_tokens = remove_tokens or []
        self.rouge = Rouge()
        self.output_dir = Path(config.general.output_path) if hasattr(config.general, 'output_path') else Path('outputs')
        self.step = 0
        
    def __call__(self, eval_preds):
        predictions, labels = eval_preds
        self.step += 1
        
        print("\n=== Metric Calculation Debug ===")
        print(f"Input shapes - Predictions: {predictions.shape}, Labels: {labels.shape}")
        
        # 음수값 처리
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = predictions.clip(min=0)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        try:
            # 디코딩 및 후처리
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # 특수 토큰 제거 및 공백 정리
            decoded_preds = [self._clean_text(pred) for pred in decoded_preds]
            decoded_labels = [self._clean_text(label) for label in decoded_labels]
            
            print("\n=== Sample Predictions ===")
            for i in range(min(3, len(decoded_preds))):
                print(f"\nSample {i}:")
                print(f"Pred: {decoded_preds[i][:100]}...")
                print(f"Gold: {decoded_labels[i][:100]}...")
            
            # 샘플별 ROUGE 점수 계산
            samples_data = []
            failed_samples = []
            for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
                try:
                    if not pred.strip() or not label.strip():
                        print(f"Warning: Empty text for sample {i}")
                        print(f"Pred: '{pred}'")
                        print(f"Gold: '{label}'")
                        continue
                        
                    scores = self.rouge.get_scores(pred, label)[0]
                    print(f"\nROUGE scores for sample {i}:")
                    print(f"ROUGE-1: {scores['rouge-1']['f']:.4f}")
                    print(f"ROUGE-2: {scores['rouge-2']['f']:.4f}")
                    print(f"ROUGE-L: {scores['rouge-l']['f']:.4f}")
                    
                    # rouge 라이브러리의 키 이름을 우리의 메트릭 이름으로 매핑
                    metric_mapping = {
                        'rouge-1': 'rouge1_f1',
                        'rouge-2': 'rouge2_f1',
                        'rouge-l': 'rougeL_f1'
                    }
                    
                    sample_data = {
                        'id': i,
                        'prediction': pred,
                        'gold_summary': label,
                    }
                    
                    # 매핑된 이름으로 메트릭 저장
                    for rouge_key, metric_name in metric_mapping.items():
                        sample_data[metric_name] = float(scores[rouge_key]['f'])
                        
                    samples_data.append(sample_data)
                except Exception as e:
                    print(f"Error calculating ROUGE for sample {i}: {str(e)}")
                    print(f"Prediction: '{pred}'")
                    print(f"Gold: '{label}'")
                    failed_samples.append(i)
                    continue
            
            if failed_samples:
                print(f"\nFailed to calculate ROUGE for {len(failed_samples)} samples")
                print(f"Failed sample indices: {failed_samples}")
            
            # DataFrame 생성
            df = pd.DataFrame(samples_data)
            
            # 평균 ROUGE 점수 계산 (Trainer용 - eval_ prefix 없이)
            metrics = {
                'rouge1_f1': float(df['rouge1_f1'].mean()),
                'rouge2_f1': float(df['rouge2_f1'].mean()),
                'rougeL_f1': float(df['rougeL_f1'].mean())
            }
            
            print("\n=== ROUGE Scores ===")
            print(f"Number of valid samples: {len(df)}")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
            
            # 결과 저장
            eval_step = f"step_{self.step}"
            save_dir = self.output_dir / "eval_results" / eval_step
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 예측 결과 저장
            save_path = save_dir / "predictions_and_metrics.csv"
            df.to_csv(save_path, index=False)
            
            # wandb 로깅
            if wandb.run is not None:
                try:
                    # 테이블 생성
                    columns = ["id", "prediction", "gold_summary", "rouge1_f1", "rouge2_f1", "rougeL_f1"]
                    data = [[d[col] for col in columns] for d in samples_data]
                    table = wandb.Table(columns=columns, data=data)
                    
                    # wandb용 메트릭 (eval/ prefix 사용)
                    wandb_metrics = {
                        f"predictions_table_{eval_step}": table,
                        "eval/rouge1_f1": metrics['rouge1_f1'],
                        "eval/rouge2_f1": metrics['rouge2_f1'],
                        "eval/rougeL_f1": metrics['rougeL_f1'],
                        "eval/rouge_avg": sum(metrics.values()) / len(metrics),
                        "eval/step": self.step,
                        "eval/num_samples": len(df),
                        "eval/num_failed": len(failed_samples),
                        "eval/avg_prediction_length": df['prediction'].str.len().mean(),
                        "eval/avg_gold_length": df['gold_summary'].str.len().mean()
                    }
                    wandb.log(wandb_metrics)
                    print("\nSuccessfully logged to wandb")
                except Exception as e:
                    print(f"Error in wandb logging: {str(e)}")
                    traceback.print_exc()
            
            return metrics  # Trainer가 eval_ prefix 추가
            
        except Exception as e:
            print(f"\n=== Error in Metric Calculation ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            traceback.print_exc()
            
            # 기본값 반환 (Trainer용)
            default_metrics = {
                'rouge1_f1': 0.0,
                'rouge2_f1': 0.0,
                'rougeL_f1': 0.0
            }
            
            # wandb에 에러 로깅 (eval/ prefix 사용)
            if wandb.run is not None:
                wandb.log({
                    "eval/error": str(e),
                    "eval/step": self.step,
                    **{f"eval/{k}": v for k, v in default_metrics.items()}
                })
            
            return default_metrics
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        
        # None이 아닌 유효한 토큰만 제거
        for token in self.remove_tokens:
            if token is not None and isinstance(token, str):
                text = text.replace(token, "")
        
        return text if text else "empty" 