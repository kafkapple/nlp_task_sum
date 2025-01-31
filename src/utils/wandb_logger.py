from pathlib import Path
import wandb
import pandas as pd
from typing import Dict, List, Optional, Union
import traceback
from omegaconf import OmegaConf
from .utils import convert_to_basic_types

class WandBLogger:
    def __init__(self, config):
        self.config = config
        self.step = 0
        self.output_dir = Path(config.general.output_path)
        self._initialize_wandb()
    
    def _initialize_wandb(self):
        """WandB 초기화"""
        if wandb.run is not None:
            wandb.finish()
            
        if hasattr(self.config.general, 'wandb'):
            # OmegaConf DictConfig를 일반 dict로 변환
            config_dict = OmegaConf.to_container(self.config, resolve=True)
            # 기본 Python 타입으로 변환
            config_dict = convert_to_basic_types(config_dict)
            
            wandb.init(
                project=self.config.general.wandb.project,
                entity=self.config.general.wandb.entity,
                name=self.config.general.timestamp,
                config=config_dict,
                dir=str(self.output_dir),
                mode="online",
                reinit=True
            )
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = "eval"):
        """메트릭 로깅"""
        if wandb.run is None:
            return
            
        try:
            wandb_metrics = {
                f"{prefix}/{k}": v for k, v in metrics.items()
            }
            wandb.log(wandb_metrics, step=self.step)
        except Exception as e:
            print(f"Error logging metrics: {e}")
    
    def log_predictions(self, 
                       predictions_df: pd.DataFrame, 
                       save_path: Optional[str] = None,
                       prefix: str = "eval"):
        """예측 결과 로깅"""
        if wandb.run is None:
            return
            
        try:
            # 테이블 생성 및 로깅
            table = wandb.Table(dataframe=predictions_df)
            wandb.log({f"{prefix}/predictions": table}, step=self.step)
            
            # 파일 저장
            if save_path:
                predictions_df.to_csv(save_path, index=False)
                self.log_artifact(save_path, f"{prefix}_predictions")
                
        except Exception as e:
            print(f"Error logging predictions: {e}")
            traceback.print_exc()
    
    def log_artifact(self, file_path: str, name: str, type: str = "dataset"):
        """Artifact 로깅"""
        if wandb.run is None:
            return
            
        try:
            artifact = wandb.Artifact(
                name=name,
                type=type,
                description=f"Results from step {self.step}"
            )
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"Error logging artifact: {e}") 