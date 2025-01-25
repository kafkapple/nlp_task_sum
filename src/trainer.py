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

class CustomLoggingCallback(TrainerCallback):
    """커스텀 로깅 콜백"""
    def __init__(self, logging_steps=5):
        self.logging_steps = logging_steps
        
    def on_step_end(self, args, state, control, **kwargs):
        """스텝 단위 로깅 - 기본 메트릭만"""
        if state.global_step % self.logging_steps == 0:
            logs = {}
            # 기본 메트릭만 로깅
            for key in ['loss', 'learning_rate', 'grad_norm']:
                if key in state.log_history[-1]:
                    logs[key] = state.log_history[-1][key]
            
            if wandb.run is not None:
                wandb.log(logs, step=state.global_step)
            
            # 콘솔 출력 없음 (디버그 프린트 제거)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """에폭 단위 로깅 - 모든 메트릭"""
        if metrics:
            # Rouge 메트릭 등 모든 메트릭 로깅
            if wandb.run is not None:
                wandb.log(metrics, step=state.global_step)
            
            # 에폭 단위 콘솔 출력
            print(f"\nEpoch {state.epoch:.2f} 평가:")
            for key, value in metrics.items():
                if key.startswith('eval_'):
                    print(f"{key}: {value:.4f}")
            print("-" * 50)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        if 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        super().__init__(*args, **kwargs)
        self.tokenizer = kwargs.get('processing_class')
        # 커스텀 콜백 추가
        self.add_callback(CustomLoggingCallback(logging_steps=self.args.logging_steps))

    def training_step(self, model, inputs, return_outputs: bool = False):
        if not hasattr(self, 'optimizer_scheduler_debug_printed'):
            print("\n=== Training Step Debug ===")
            print(f"Optimizer exists: {hasattr(self, 'optimizer')}")
            print(f"Scheduler exists: {hasattr(self, 'lr_scheduler')}")
            if hasattr(self, 'optimizer'):
                print(f"Optimizer type: {type(self.optimizer)}")
            if hasattr(self, 'lr_scheduler'):
                print(f"Scheduler type: {type(self.lr_scheduler)}")
            self.optimizer_scheduler_debug_printed = True
        
        outputs = super().training_step(model, inputs, return_outputs)
        
        # step 단위로 loss와 learning rate 로깅
        if self.is_world_process_zero() and wandb.run is not None:
            try:
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
                # lr_scheduler가 None이 아닌 경우에만 learning rate 로깅
                if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                    lr = self.lr_scheduler.get_last_lr()[0]
                else:
                    lr = self.args.learning_rate
                
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": lr,
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
        
        # 생성 설정
        gen_kwargs = {
            "max_new_tokens": self.args.generation_max_length,  # generation_max_length를 max_new_tokens로 사용
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

    def create_optimizer(self):
        """커스텀 옵티마이저 생성"""
        print("\n=== Creating Optimizer ===")
        try:
            optimizer = super().create_optimizer()
            print(f"Optimizer type: {type(optimizer)}")
            if optimizer is None:
                print("Creating fallback AdamW optimizer")
                from transformers import AdamW
                optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay
                )
            return optimizer
        except Exception as e:
            print(f"Warning: Failed to create optimizer: {e}")
            print("Creating default AdamW optimizer")
            from torch.optim import AdamW
            return AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """커스텀 스케줄러 생성"""
        print("\n=== Creating Scheduler ===")
        print(f"num_training_steps: {num_training_steps}")
        print(f"input optimizer: {type(optimizer) if optimizer else None}")
        print(f"self.optimizer exists: {hasattr(self, 'optimizer')}")
        
        try:
            # optimizer 확인 및 생성
            if optimizer is None:
                print("No optimizer provided, checking self.optimizer")
                if not hasattr(self, 'optimizer'):
                    print("Creating new optimizer")
                    self.optimizer = self.create_optimizer()
                optimizer = self.optimizer
            
            print(f"Final optimizer type: {type(optimizer)}")
            print(f"Scheduler type: {self.args.lr_scheduler_type}")
            
            # 부모 클래스의 create_scheduler 먼저 호출
            scheduler = super().create_scheduler(num_training_steps, optimizer=optimizer)
            
            # 부모 클래스의 스케줄러가 None인 경우에만 커스텀 스케줄러 생성
            if scheduler is None:
                if self.args.lr_scheduler_type == "cosine":
                    from transformers import get_cosine_schedule_with_warmup
                    scheduler = get_cosine_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=self.args.warmup_steps or int(num_training_steps * self.args.warmup_ratio),
                        num_training_steps=num_training_steps,
                    )
                    print(f"Created cosine scheduler: {type(scheduler)}")
                else:
                    from transformers import get_linear_schedule_with_warmup
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=self.args.warmup_steps or int(num_training_steps * self.args.warmup_ratio),
                        num_training_steps=num_training_steps,
                    )
                    print(f"Created linear scheduler: {type(scheduler)}")
            
            # accelerate와의 호환성을 위해 scheduler 속성 직접 설정
            if hasattr(self, 'lr_scheduler'):
                print("Updating existing lr_scheduler")
                self.lr_scheduler = scheduler
            
            print(f"Final scheduler type: {type(scheduler)}")
            return scheduler
        
        except Exception as e:
            print(f"Error in create_scheduler: {str(e)}")
            raise  # 에러를 다시 발생시켜 디버깅 용이하게 