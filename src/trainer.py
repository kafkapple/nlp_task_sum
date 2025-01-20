from transformers import Seq2SeqTrainer, TrainerCallback
from typing import Dict, Any
import wandb

class WandBCallback(TrainerCallback):
    """WandB 로깅을 위한 커스텀 콜백"""
    def __init__(self):
        self.training_config = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        """학습 시작 시 설정을 안전하게 변환해서 로깅"""
        if self.training_config is None:
            # 기본 타입만 포함하도록 설정 필터링
            self.training_config = {
                k: v for k, v in vars(args).items() 
                if isinstance(v, (int, float, str, bool, type(None)))
            }
            wandb.config.update({"training": self.training_config}, allow_val_change=True)
    
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

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, remove_tokens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.remove_tokens = remove_tokens or []
        self.wandb_callback = WandBCallback()
        self.add_callback(self.wandb_callback)
    
    def compute_metrics(self, pred):
        """compute_metrics 래퍼 함수"""
        return self.args.compute_metrics(pred)  # 직접 로깅 제거 