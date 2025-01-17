from transformers import Seq2SeqTrainer
from typing import Dict
import wandb

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        # model 객체에서 내부 모델을 추출
        if hasattr(kwargs['model'], 'model'):
            kwargs['model'] = kwargs['model'].model
        super().__init__(*args, **kwargs)
        
    def log(self, logs: Dict[str, float], iterator=None, start_time=None) -> None:
        """로깅 커스터마이징"""
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

        wandb.log(logs) 