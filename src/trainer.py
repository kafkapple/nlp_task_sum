from transformers import Trainer, TrainerCallback, GenerationConfig, PreTrainedTokenizerBase
from typing import Dict, List, Optional, Union
import wandb
import torch
import numpy as np
from rouge import Rouge  # 원본 baseline 방식으로 변경
from pathlib import Path
import os

class WandBCallback(TrainerCallback):
    """WandB 로깅을 위한 콜백"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """로그 이벤트 처리"""
        if state.is_world_process_zero and logs:
            wandb.log(logs)
            
    def on_train_begin(self, args, state, control, **kwargs):
        """학습 시작 시 처리"""
        if state.is_world_process_zero:
            wandb.define_metric("train/global_step")
            wandb.define_metric("*", step_metric="train/global_step", step_sync=True)
            
    def on_train_end(self, args, state, control, **kwargs):
        """학습 종료 시 처리"""
        if state.is_world_process_zero:
            wandb.finish()

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

class CustomTrainer(Trainer):
    def __init__(self, *args, remove_tokens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.remove_tokens = remove_tokens or []
        
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """오버라이드된 prediction_step"""
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        
        # 생성 설정
        gen_kwargs = {
            "max_length": self.args.generation_max_length,
            "num_beams": self.args.generation_num_beams,
            "synced_gpus": True if os.getenv("ACCELERATE_TORCH_DISTRIBUTED_DEBUG") else False,
        }
        
        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        
        # labels가 있으면 loss 계산
        if has_labels:
            with torch.no_grad():
                outputs = model(**inputs)
            loss = outputs.loss.mean().detach()
        else:
            loss = None
            
        return (loss, generated_tokens, inputs["labels"])

    def compute_metrics(self, eval_preds):
        """메트릭 계산 및 로깅"""
        # 부모 클래스의 compute_metrics 속성 사용
        metrics = super().compute_metrics(eval_preds)
        
        # wandb에 메트릭 직접 로깅 (이미 eval/ 접두사가 있으므로 추가하지 않음)
        if wandb.run is not None:
            wandb.log(metrics)
            
        return metrics 