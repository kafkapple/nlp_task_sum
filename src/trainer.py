from transformers import Seq2SeqTrainer, TrainerCallback, GenerationConfig, PreTrainedTokenizerBase
from typing import Dict, Any
import wandb
import torch
import numpy as np
from rouge import Rouge  # 원본 baseline 방식으로 변경

class WandBCallback(TrainerCallback):
    """WandB 로깅을 위한 커스텀 콜백"""
    def __init__(self):
        self.training_config = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        """학습 시작 시 설정을 안전하게 변환해서 로깅"""
        if self.training_config is None:
            # GPU 정보 수집
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
            }
            
            # 기본 타입만 포함하도록 설정 필터링
            self.training_config = {
                k: v for k, v in vars(args).items() 
                if isinstance(v, (int, float, str, bool, type(None)))
            }
            
            # 설정 업데이트
            wandb.config.update({
                "training": self.training_config,
                "hardware": gpu_info
            }, allow_val_change=True)
    
    def on_log(self, args, state, control, logs: Dict[str, Any] = None, **kwargs):
        if not logs:
            return
            
        clean_logs = {}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                # eval_ 접두사를 eval/로 변환하여 wandb에 로깅
                if k.startswith('eval_'):
                    clean_logs[k.replace('eval_', 'eval/')] = v
                else:
                    clean_logs[f'train/{k}'] = v
        
        if state.epoch is not None:
            clean_logs["epoch"] = round(state.epoch, 2)
        if state.global_step is not None:
            clean_logs["step"] = state.global_step
            
        wandb.log(clean_logs)

class TrainerMetrics:
    def __init__(self, tokenizer, remove_tokens=None):
        self.processing_class = tokenizer
        self.remove_tokens = remove_tokens or []
        self.rouge = Rouge()
        
    def __call__(self, eval_preds):
        predictions, labels = eval_preds
        
        print("\n=== Metric Calculation Debug ===")
        print(f"Input shapes - Predictions: {predictions.shape}, Labels: {labels.shape}")
        
        # 음수값 처리
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = predictions.clip(min=0)
        labels = np.where(labels != -100, labels, self.processing_class.pad_token_id)
        
        print(f"\nToken ranges:")
        print(f"Predictions: [{predictions.min()}, {predictions.max()}]")
        print(f"Labels: [{labels.min()}, {labels.max()}]")
        
        try:
            # 디코딩
            decoded_preds = self.processing_class.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=True)
            
            # 샘플 출력 (처음 3개)
            print("\n=== Sample Outputs ===")
            for i in range(min(3, len(decoded_preds))):
                print(f"\n[Sample {i+1}]")
                print(f"Prediction tokens: {predictions[i][:50]}...")
                print(f"Label tokens: {labels[i][:50]}...")
                print(f"Decoded prediction: {decoded_preds[i][:200]}")
                print(f"Decoded label: {decoded_labels[i][:200]}")
                print("-" * 80)
            
            # 특수 토큰 제거 및 공백 정리
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]
            
            for token in self.remove_tokens:
                decoded_preds = [pred.replace(token, "") for pred in decoded_preds]
                decoded_labels = [label.replace(token, "") for label in decoded_labels]
            
            # 빈 문자열 처리 및 통계
            empty_preds = sum(1 for p in decoded_preds if not p.strip())
            empty_labels = sum(1 for l in decoded_labels if not l.strip())
            
            print(f"\n=== Empty String Statistics ===")
            print(f"Empty predictions: {empty_preds}/{len(decoded_preds)}")
            print(f"Empty labels: {empty_labels}/{len(decoded_labels)}")
            
            # 빈 문자열 대체
            decoded_preds = [pred if pred.strip() else "empty" for pred in decoded_preds]
            decoded_labels = [label if label.strip() else "empty" for label in decoded_labels]
            
            # 길이 통계
            pred_lengths = [len(p.split()) for p in decoded_preds]
            label_lengths = [len(l.split()) for l in decoded_labels]
            
            print(f"\n=== Length Statistics ===")
            print(f"Prediction lengths (min/max/avg): {min(pred_lengths)}/{max(pred_lengths)}/{np.mean(pred_lengths):.1f}")
            print(f"Label lengths (min/max/avg): {min(label_lengths)}/{max(label_lengths)}/{np.mean(label_lengths):.1f}")
            
            # ROUGE 계산
            scores = self.rouge.get_scores(decoded_preds, decoded_labels, avg=True)
            
            result = {
                'rouge1_f1': scores['rouge-1']['f'],
                'rouge2_f1': scores['rouge-2']['f'],
                'rougeL_f1': scores['rouge-l']['f']
            }
            
            print("\n=== Final ROUGE Scores ===")
            print(result)
            
            return result
            
        except Exception as e:
            print(f"\n=== Error in Metric Calculation ===")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            import traceback
            print("\nTraceback:")
            print(traceback.format_exc())
            
            return {
                'rouge1_f1': 0.0,
                'rouge2_f1': 0.0,
                'rougeL_f1': 0.0
            }

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, remove_tokens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.remove_tokens = remove_tokens or []
        
        # tokenizer를 processing_class로 설정
        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            self.processing_class = self.tokenizer
        
        # 모델의 특수 토큰 ID 저장
        self._save_model_special_tokens()
        
    def _save_model_special_tokens(self):
        """모델의 특수 토큰 ID를 저장"""
        self.special_tokens = {
            "bos_token_id": self.processing_class.bos_token_id,
            "eos_token_id": self.processing_class.eos_token_id,
            "pad_token_id": self.processing_class.pad_token_id
        }
        
        # 모델 config에도 설정
        self.model.config.decoder_start_token_id = self.special_tokens["bos_token_id"]
        self.model.config.forced_bos_token_id = self.special_tokens["bos_token_id"]
        self.model.config.forced_eos_token_id = self.special_tokens["eos_token_id"]
        
        # Generation config 설정
        self.model.generation_config = GenerationConfig(
            decoder_start_token_id=self.special_tokens["bos_token_id"],
            bos_token_id=self.special_tokens["bos_token_id"],
            eos_token_id=self.special_tokens["eos_token_id"],
            forced_bos_token_id=self.special_tokens["bos_token_id"],
            forced_eos_token_id=self.special_tokens["eos_token_id"],
            pad_token_id=self.special_tokens["pad_token_id"]
        )
        
    def prediction_step(self, *args, **kwargs):
        """생성 시 특수 토큰 ID 유지"""
        # 생성 설정 재확인
        self._save_model_special_tokens()
        return super().prediction_step(*args, **kwargs)

    def compute_metrics(self, eval_preds):
        """메트릭 계산 및 로깅"""
        metrics = self.args.compute_metrics(eval_preds)
        
        # wandb에 메트릭 직접 로깅 (이미 eval/ 접두사가 있으므로 추가하지 않음)
        if wandb.run is not None:
            wandb.log(metrics)
            
        return metrics 