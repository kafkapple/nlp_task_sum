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
        metrics = super().compute_metrics(eval_preds)
        
        # wandb에 메트릭 로깅 시 eval/ 접두사 추가
        if wandb.run is not None:
            logged_metrics = {
                f"eval/{k}" if not k.startswith('eval/') else k: v 
                for k, v in metrics.items()
            }
            wandb.log(logged_metrics)
            
        return metrics 