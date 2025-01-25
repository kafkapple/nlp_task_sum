import pandas as pd
from pathlib import Path
from typing import List
import os
import wandb
from omegaconf import DictConfig, OmegaConf 


def save_predictions(predictions: List[str], fnames: List[str], output_path: str):
    """Save predictions to CSV file"""
    output = pd.DataFrame({
        "fname": fnames,
        "summary": predictions
    })
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False) 


def convert_to_basic_types(obj):
    """OmegaConf 객체를 기본 Python 타입으로 변환"""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)) or hasattr(obj, '_type'):  # ListConfig 포함
        return [convert_to_basic_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_basic_types(v) for k, v in obj.items()}
    else:
        return str(obj)

def init_wandb(cfg):
    """wandb 초기화"""
    try:
        if wandb.run is not None:
            wandb.finish()
            
        output_dir = Path(cfg['general']['output_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if 'wandb' in cfg['general']:
            wandb.init(
                project=cfg['general']['wandb']['project'],
                name=cfg['general']['timestamp'],
                config=cfg,  # 이미 변환된 config 사용
                dir=str(output_dir),
                mode="online",
                reinit=True
            )
        
        return output_dir
        
    except Exception as e:
        print(f"Warning: wandb initialization failed: {str(e)}")
        return Path(cfg['general']['output_path'])

def get_model_size(model):
    """모델의 파라미터 수를 반환"""
    return sum(p.numel() for p in model.parameters())

def print_samples(original_texts: List[str], 
                 gold_summaries: List[str], 
                 pred_summaries: List[str], 
                 n_samples: int = 3):
    """샘플 텍스트 출력 (원본, 정답, 예측)"""
    print("\n" + "="*50 + " Samples " + "="*50)
    
    for i in range(min(n_samples, len(original_texts))):
        print(f"\n[Sample {i+1}]")
        print(f"\nOriginal Text:\n{original_texts[i]}")
        print(f"\nGold Summary:\n{gold_summaries[i]}")
        print(f"\nPredicted Summary:\n{pred_summaries[i]}")
        print("\n" + "="*110)

def setup_seeds(seed: int):
    """모든 random seed 설정"""
    import random
    import numpy as np
    import pandas as pd
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pd.options.mode.chained_assignment = None