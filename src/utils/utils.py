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

def init_wandb(cfg: DictConfig):
    """wandb 초기화"""
    try:
        # 이전 wandb 프로세스 정리
        if wandb.run is not None:
            wandb.finish()
            
        timestamp = cfg.general.timestamp
        run_name = f"{cfg.model.name}_{cfg.model.mode}_{timestamp}"
        
        # output_path 생성
        output_dir = Path(cfg.general.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # config 폴더 복사
        import shutil
        config_dir = Path("config")
        output_config_dir = output_dir / "config"
        if config_dir.exists():
            if output_config_dir.exists():
                shutil.rmtree(output_config_dir)
            shutil.copytree(config_dir, output_config_dir)
            print(f"Copied config files to {output_config_dir}")
        
        # wandb 설정
        os.environ["WANDB_DIR"] = str(output_dir)
        os.environ["WANDB_START_METHOD"] = "thread"
        os.environ["WANDB_WATCH"] = "false"  # 모델 가중치 로깅 비활성화
        
        # 설정을 기본 Python 타입으로 변환
        config_dict = convert_to_basic_types(OmegaConf.to_container(cfg, resolve=True))
        
        # wandb 초기화
        if hasattr(cfg.general, 'wandb'):
            wandb.init(
                project=cfg.general.wandb.project,
                name=run_name,
                config=config_dict,
                group=cfg.model.family,
                dir=str(output_dir),
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True,
                    _disable_meta=True
                ),
                mode="online",
                reinit=True
            )
        else:
            print("Warning: wandb configuration not found in config. Running without wandb logging.")
        
        return output_dir
        
    except Exception as e:
        print(f"Warning: wandb initialization failed: {str(e)}")
        print("Continuing without wandb logging...")
        return Path(cfg.general.output_path)

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