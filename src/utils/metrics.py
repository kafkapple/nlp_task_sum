from rouge import Rouge
from typing import Dict, List
import wandb
import numpy as np
from transformers import PreTrainedTokenizerBase
from omegaconf import DictConfig
from pathlib import Path

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
        self.output_dir = Path(config.general.output_dir) if hasattr(config.general, 'output_dir') else Path('outputs')
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
            print(f"\nToken ranges:")
            print(f"Predictions: [{predictions.min()}, {predictions.max()}]")
            print(f"Labels: [{labels.min()}, {labels.max()}]")
            # 샘플별 ROUGE 점수 계산 및 결과 저장
            samples_data = []
            for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
                try:
                    scores = self.rouge.get_scores(pred, label)[0]
                    sample_data = {
                        'id': i,
                        'prediction': pred,
                        'gold_summary': label,
                        'rouge-1': scores['rouge-1']['f'],
                        'rouge-2': scores['rouge-2']['f'],
                        'rouge-l': scores['rouge-l']['f']
                    }
                    samples_data.append(sample_data)
                except Exception as e:
                    print(f"Error calculating ROUGE for sample {i}: {str(e)}")
                    continue
            
            # DataFrame 생성 및 저장
            import pandas as pd
            df = pd.DataFrame(samples_data)
            
            # 평균 ROUGE 점수 계산
            result = {
                'rouge1_f1': df['rouge-1'].mean(),
                'rouge2_f1': df['rouge-2'].mean(),
                'rougeL_f1': df['rouge-l'].mean()
            }
            
            # 결과 저장 경로 확실히 생성
            eval_step = f"step_{self.step}"
            save_dir = self.output_dir / "eval_results" / eval_step
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # CSV 파일로 저장 시 경로 확인 로그 추가
            save_path = save_dir / "predictions_and_metrics.csv"
            print(f"\nSaving predictions to: {save_path}")
            df.to_csv(save_path, index=False)
            
            # 통계 정보 저장
            stats = {
                'num_samples': len(df),
                'avg_prediction_length': df['prediction'].str.len().mean(),
                'avg_gold_length': df['gold_summary'].str.len().mean(),
                'rouge_scores': result
            }
            
            # wandb 로깅 수정
            if wandb.run is not None:
                # 테이블 생성
                columns = ["id", "prediction", "gold_summary", "rouge-1", "rouge-2", "rouge-l"]
                data = [[d[col] for col in columns] for d in samples_data]
                table = wandb.Table(columns=columns, data=data)
                
                wandb.log({
                    f"eval/predictions_table_{eval_step}": table,
                    "eval/rouge1_f1": result['rouge1_f1'],  # eval/ 접두사로 수정
                    "eval/rouge2_f1": result['rouge2_f1'],
                    "eval/rougeL_f1": result['rougeL_f1'],
                    "eval/stats/num_samples": stats['num_samples'],
                    "eval/stats/avg_prediction_length": stats['avg_prediction_length'],
                    "eval/stats/avg_gold_length": stats['avg_gold_length']
                })
            
            # eval_ 접두사 없이 반환
            return result
            
        except Exception as e:
            print(f"\n=== Error in Metric Calculation ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'rouge1_f1': 0.0,
                'rouge2_f1': 0.0,
                'rougeL_f1': 0.0
            }
            
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        text = text.strip()
        for token in self.remove_tokens:
            text = text.replace(token, "")
        return text if text else "empty" 