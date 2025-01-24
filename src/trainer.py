from transformers import Trainer, TrainerCallback
from typing import Dict, List, Optional, Union, Any
import wandb
import torch
import numpy as np
from pathlib import Path
import os
import torch.nn as nn
from rouge import Rouge
import pandas as pd

class WandBCallback(TrainerCallback):
    """WandB 로깅을 위한 콜백"""
    def __init__(self):
        self._initialized = False
        
    def setup(self, args, state, model):
        """초기 설정"""
        if not self._initialized and state.is_world_process_zero:
            # 메트릭 정의
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step")
            wandb.define_metric("eval/*", step_metric="train/global_step")
            # 새로운 메트릭 정의
            wandb.define_metric("train/loss", summary="min")
            wandb.define_metric("train/learning_rate", summary="last")
            wandb.define_metric("eval/loss", summary="min")
            wandb.define_metric("train/rouge*", summary="max")
            wandb.define_metric("eval/rouge*", summary="max")
            self._initialized = True
    
    def on_train_begin(self, args, state, control, **kwargs):
        """학습 시작 시 처리"""
        self.setup(args, state, kwargs.get('model'))
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """로그 이벤트 처리"""
        if state.is_world_process_zero and logs:
            logged_logs = {}
            for k, v in logs.items():
                if k == "epoch":
                    logged_logs[k] = v
                elif k == "loss":
                    logged_logs["train/loss"] = v  # 학습 손실
                elif k == "learning_rate":
                    logged_logs["train/learning_rate"] = v  # 학습률
                elif not k.startswith(("train/", "eval/")):
                    logged_logs[f"train/{k}"] = v
                else:
                    logged_logs[k] = v
            
            if wandb.run is not None:
                wandb.log(logged_logs, step=state.global_step)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """평가 단계에서의 로깅"""
        if state.is_world_process_zero and metrics:
            # eval/ 접두사 추가
            logged_metrics = {}
            for k, v in metrics.items():
                metric_name = k if k.startswith('eval/') else f'eval/{k}'
                logged_metrics[metric_name] = v
            
            if wandb.run is not None:
                wandb.log(logged_metrics, step=state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        """학습 종료 시 처리"""
        if state.is_world_process_zero and wandb.run is not None:
            wandb.finish()

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        if 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        super().__init__(*args, **kwargs)
        self.tokenizer = kwargs.get('processing_class')

    def training_step(self, model, inputs, return_outputs: bool = False):
        outputs = super().training_step(model, inputs, return_outputs)
        
        # loss 로깅
        if self.is_world_process_zero() and wandb.run is not None:
            try:
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                }, step=self.state.global_step)
            except Exception as e:
                print(f"Warning: Failed to log metrics: {e}")
        
        return outputs

    def evaluation_step(self, model, inputs, prediction_loss_only=None):
        outputs = super().evaluation_step(model, inputs, prediction_loss_only)
        
        # 평가 결과 로깅
        if self.is_world_process_zero() and wandb.run is not None:
            try:
                wandb.log({
                    "eval/loss": outputs.loss.item(),
                }, step=self.state.global_step)
            except Exception as e:
                print(f"Warning: Failed to log eval metrics: {e}")
        
        return outputs

    def compute_rouge_scores(self, preds, labels):
        """ROUGE 점수 계산"""
        rouge = Rouge()
        scores = rouge.get_scores(preds, labels, avg=True)
        return scores

    def compute_metrics(self, eval_preds):
        """메트릭 계산 및 로깅"""
        if not hasattr(self, "compute_metrics_fn"):
            return {}
            
        metrics = self.compute_metrics_fn(eval_preds)
        
        # 예측 결과 디코딩
        predictions = self.tokenizer.batch_decode(
            eval_preds.predictions, skip_special_tokens=True
        )
        references = self.tokenizer.batch_decode(
            eval_preds.label_ids, skip_special_tokens=True
        )
        
        # 예측 결과 저장
        eval_step = f"step_{self.state.global_step}"
        save_dir = Path(self.args.output_dir) / "eval_results" / eval_step
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 결과를 DataFrame으로 변환
        results_df = pd.DataFrame({
            "prediction": predictions,
            "reference": references,
            "rouge1": metrics.get("rouge1_f1", [0] * len(predictions)),
            "rouge2": metrics.get("rouge2_f1", [0] * len(predictions)),
            "rougeL": metrics.get("rougeL_f1", [0] * len(predictions))
        })
        
        # CSV 파일로 저장
        save_path = save_dir / "predictions.csv"
        results_df.to_csv(save_path, index=False)
        print(f"\nSaved predictions to: {save_path}")
        
        # WandB 테이블 로깅
        if self.is_world_process_zero() and wandb.run is not None:
            try:
                table = wandb.Table(dataframe=results_df)
                wandb.log({
                    f"eval/predictions_table_{eval_step}": table,
                    **{f"eval/{k}": v for k, v in metrics.items()}
                }, step=self.state.global_step)
            except Exception as e:
                print(f"Warning: Failed to log predictions table: {e}")
        
        return metrics

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """오버라이드된 prediction_step"""
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        
        # 생성 설정 - max_length 대신 max_new_tokens 사용
        gen_kwargs = {
            "max_new_tokens": self.args.generation_max_length,  # max_length 대신 max_new_tokens 사용
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