from transformers import Trainer, TrainerCallback
from typing import Dict, List, Optional, Union, Any
import wandb
import torch
import numpy as np
from pathlib import Path
import os
import torch.nn as nn

class WandBCallback(TrainerCallback):
    """WandB 로깅을 위한 콜백"""
    def __init__(self):
        self._initialized = False
        
    def setup(self, args, state, model):
        """초기 설정"""
        if not self._initialized and state.is_world_process_zero:
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step")
            wandb.define_metric("eval/*", step_metric="train/global_step")
            self._initialized = True
    
    def on_train_begin(self, args, state, control, **kwargs):
        """학습 시작 시 처리"""
        self.setup(args, state, kwargs.get('model'))
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """로그 이벤트 처리"""
        if state.is_world_process_zero and logs:
            # train/ 접두사 추가
            logged_logs = {}
            for k, v in logs.items():
                if k == "epoch":
                    logged_logs[k] = v
                elif k == "loss":
                    logged_logs["train/loss"] = v
                elif k == "learning_rate":
                    logged_logs["train/learning_rate"] = v
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
        self._wandb_initialized = False

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], return_loss=True, return_outputs=False):
        """학습 스텝 최적화"""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            if return_outputs:
                outputs = model(**inputs)
                loss = outputs.loss
            else:
                loss = self.compute_loss(model, inputs)
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        loss.backward()
        
        if return_outputs:
            return loss.detach(), outputs
        return loss.detach()

    def compute_metrics(self, eval_preds):
        """메트릭 계산"""
        if not hasattr(self, "compute_metrics_fn"):
            return {}
            
        metrics = self.compute_metrics_fn(eval_preds)
        
        # eval/ 접두사 추가
        metrics = {
            f"eval/{k}" if not k.startswith('eval/') else k: v 
            for k, v in metrics.items()
        }
        
        # wandb에 직접 로깅
        if self.is_world_process_zero() and wandb.run is not None:
            wandb.log(metrics, step=self.state.global_step)
        
        return metrics

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