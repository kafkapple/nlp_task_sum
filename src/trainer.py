from transformers import Seq2SeqTrainer, TrainerCallback
from typing import Dict, Any

class WandBCallback(TrainerCallback):
    """WandB 로깅을 위한 커스텀 콜백"""
    def on_log(self, args, state, control, logs: Dict[str, Any] = None, **kwargs):
        if not logs:
            return
        
        # 기본 메트릭 처리
        clean_logs = {}
        for k, v in logs.items():
            try:
                if isinstance(v, (int, float, str, bool)):
                    clean_logs[k] = v
                else:
                    # 복잡한 객체는 문자열로 변환
                    clean_logs[k] = str(v)
            except:
                continue
                
        # epoch 추가
        if state.epoch is not None:
            clean_logs["epoch"] = round(state.epoch, 2)
            
        # step 추가
        if state.global_step is not None:
            clean_logs["step"] = state.global_step
            
        # wandb에 로깅
        import wandb
        wandb.log(clean_logs)

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(WandBCallback()) 