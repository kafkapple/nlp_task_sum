from transformers import Trainer, TrainerCallback, GenerationConfig, PreTrainedTokenizerBase
from typing import Dict, List, Optional, Union
import wandb
import torch
import numpy as np
from rouge import Rouge  # 원본 baseline 방식으로 변경
from pathlib import Path
import os
from src.utils.metrics import TrainerMetrics  # 메트릭 클래스 import

class WandBCallback(TrainerCallback):
    """WandB 로깅을 위한 콜백"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """로그 이벤트 처리"""
        if state.is_world_process_zero and logs:
            # step 정보 추가
            if state.global_step is not None:
                logs["train/global_step"] = state.global_step
            
            # loss와 learning rate 로깅
            if "loss" in logs:
                logs["train/loss"] = logs.pop("loss")
            if "learning_rate" in logs:
                logs["train/learning_rate"] = logs.pop("learning_rate")
                
            wandb.log(logs, step=state.global_step)
            
    def on_train_begin(self, args, state, control, **kwargs):
        """학습 시작 시 처리"""
        if state.is_world_process_zero:
            # 메트릭 정의
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step")
            wandb.define_metric("eval/*", step_metric="train/global_step")
            
    def on_train_end(self, args, state, control, **kwargs):
        """학습 종료 시 처리"""
        if state.is_world_process_zero:
            wandb.finish()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """평가 단계 처리"""
        if state.is_world_process_zero and metrics:
            # eval 메트릭 로깅
            eval_metrics = {}
            for k, v in metrics.items():
                metric_name = k if k.startswith('eval/') else f'eval/{k}'
                eval_metrics[metric_name] = v
            
            wandb.log(eval_metrics, step=state.global_step)

class CustomTrainer(Trainer):
    def __init__(self, *args, remove_tokens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.remove_tokens = remove_tokens or []
        
    def log(self, logs: Dict[str, float]) -> None:
        """로깅 오버라이드"""
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 4)

        if "learning_rate" in logs:
            logs["train/learning_rate"] = logs.pop("learning_rate")
        if "loss" in logs:
            logs["train/loss"] = logs.pop("loss")
            
        output = {**logs}
        if self.is_world_process_zero():
            self.log_metrics("train", output)
            self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

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
        metrics = self.compute_metrics_fn(eval_preds)
        
        # wandb에 메트릭 로깅
        if self.is_world_process_zero() and wandb.run is not None:
            logged_metrics = {}
            for k, v in metrics.items():
                # eval/ 접두사가 없는 경우에만 추가
                metric_name = k if k.startswith('eval/') else f'eval/{k}'
                logged_metrics[metric_name] = v
                
            wandb.log(logged_metrics, step=self.state.global_step)
            
        return metrics 