import hydra
from omegaconf import DictConfig
import warnings
import torchvision
from typing import List
import pandas as pd
import os
import numpy as np
import random

# Beta transforms 경고 끄기
torchvision.disable_beta_transforms_warning()

# 다른 경고들도 필요한 경우 끄기
warnings.filterwarnings("ignore", category=UserWarning)
# 또는 특정 메시지만
warnings.filterwarnings("ignore", message=".*beta.*")

from pathlib import Path
import wandb
from transformers import Seq2SeqTrainingArguments
import pytorch_lightning as pl
from datetime import datetime
from omegaconf import OmegaConf

from src.data.dataset import DataProcessor, download_and_extract, load_dataset
from src.models.model_factory import ModelFactory
from src.utils.utils import save_predictions
from src.utils.metrics import Metrics, compute_metrics
from src.trainer import CustomTrainer

def get_model_size(model):
    """모델의 파라미터 수를 반환"""
    return sum(p.numel() for p in model.parameters())

def init_wandb(cfg: DictConfig):
    """wandb 초기화"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{cfg.model.name}_{cfg.model.mode}_{timestamp}"
    
    # output_path 생성
    output_dir = Path(cfg.general.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # wandb 설정
    os.environ["WANDB_DIR"] = str(output_dir)
    
    def convert_to_basic_types(obj):
        if isinstance(obj, (int, float, str, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_basic_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_basic_types(v) for k, v in obj.items()}
        elif hasattr(obj, '_type'):  # OmegaConf 객체
            return convert_to_basic_types(OmegaConf.to_container(obj, resolve=True))
        else:
            return str(obj)
    
    # 설정을 기본 Python 타입으로 변환
    config_dict = convert_to_basic_types(OmegaConf.to_container(cfg, resolve=True))
    
    wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        config=config_dict,
        group=cfg.model.family,
        dir=str(output_dir),
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=True
        )
    )
    
    print(f"Outputs will be saved to: {output_dir}")
    return output_dir

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

@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    # 모든 random seed 설정
    setup_seeds(cfg.general.seed)
    pl.seed_everything(cfg.general.seed)
    
    # output_dir 초기화
    output_dir = init_wandb(cfg)
    
    # Ensure data directory exists
    download_and_extract(cfg.url.data, cfg.general.data_path)
    
    print("모델 생성 시작...")
    model = ModelFactory.create_model(cfg)
    print("모델 생성 완료")
    
    if hasattr(model, 'model'):
        print(f"모델 파라미터 수: {get_model_size(model.model):,}")
        wandb.config.update({
            "model/num_parameters": int(get_model_size(model.model))
        }, allow_val_change=True)
    
    if cfg.model.mode == "prompt":
        print("프롬프트 모드로 실행 중...")
        processor = DataProcessor(None, cfg)
        print("데이터 로드 중...")
        train_df, val_df, test_df = processor.load_data()
        print(f"데이터 로드 완료 (train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)})")
        
        # few-shot 샘플 준비
        sample_dialogue = None
        sample_summary = None
        
        if cfg.prompt.mode == "few-shot":
            print("Few-shot 샘플 준비 중...")
            # pandas의 sample 메서드에 random_state 전달
            few_shot_samples = train_df.sample(
                n=cfg.prompt.n_samples, 
                random_state=cfg.general.seed  # 여기도 seed 전달
            )
            sample_dialogue = few_shot_samples.iloc[0]['dialogue']
            sample_summary = few_shot_samples.iloc[0]['summary']
            print(f"\nFew-shot 샘플:")
            print(f"대화: {sample_dialogue}")
            print(f"요약: {sample_summary}\n")
        
        # Initialize metrics
        metrics = Metrics()
        
        print(f"\nTesting prompt version {cfg.prompt.version}")
        # 검증 데이터 요약 생성 (한 번만 실행)
        print("검증 데이터 요약 생성 중...")
        val_predictions = model.batch_summarize(
            val_df['dialogue'].tolist()[:cfg.prompt.n_samples],
            sample_dialogue,
            sample_summary,
            prompt_version=cfg.prompt.version
        )
        
        # 메트릭 계산 및 로깅
        val_score, avg_score_dict = metrics.compute_batch_metrics(
            val_predictions,
            val_df['summary'].tolist()[:cfg.prompt.n_samples]
        )
        print(f"\n===Validation Score - Average: {val_score}\n{avg_score_dict}")
        key_map = cfg.metrics
        wandb.log({key_map[k]: v for k, v in avg_score_dict.items() if k in key_map})
        
        if cfg.prompt.save_predictions:
            print("테스트 데이터 요약 생성 중...")
            test_predictions = model.batch_summarize(
                test_df['dialogue'].tolist(),
                sample_dialogue,
                sample_summary,
                prompt_version=cfg.prompt.version
            )
            save_predictions(
                test_predictions,
                test_df['fname'].tolist(),
                output_dir / "prediction_prompt.csv"
            )
            
        # 샘플 출력
        print_samples(
            val_df['dialogue'].tolist()[:cfg.prompt.n_samples],
            val_df['summary'].tolist()[:cfg.prompt.n_samples],
            val_predictions
        )
        
    else:  # finetune 모드
        processor = DataProcessor(model.tokenizer, cfg)
        train_dataset = processor.prepare_dataset("train")
        val_dataset = processor.prepare_dataset("dev")
        
        # Training arguments 설정
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            learning_rate=cfg.train.training.learning_rate,
            num_train_epochs=cfg.train.training.num_train_epochs,
            per_device_train_batch_size=cfg.train.training.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.train.training.per_device_eval_batch_size,
            warmup_ratio=cfg.train.training.warmup_ratio,
            weight_decay=cfg.train.training.weight_decay,
            fp16=cfg.train.training.fp16,
            evaluation_strategy=cfg.train.training.evaluation_strategy,
            save_strategy=cfg.train.training.save_strategy,
            save_total_limit=cfg.train.training.save_total_limit,
            load_best_model_at_end=cfg.train.training.load_best_model_at_end,
            metric_for_best_model=cfg.train.training.metric_for_best_model,
            greater_is_better=cfg.train.training.greater_is_better,
            gradient_checkpointing=cfg.train.training.gradient_checkpointing,
            gradient_accumulation_steps=cfg.train.training.gradient_accumulation_steps,
            predict_with_generate=cfg.train.training.predict_with_generate,
            generation_max_length=cfg.train.training.generation_max_length,
            generation_num_beams=cfg.train.training.generation_num_beams,
            remove_unused_columns=cfg.train.training.remove_unused_columns
        )
        
        trainer = CustomTrainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=model.tokenizer,
            compute_metrics=compute_metrics
        )
        
        trainer.train()
    
    wandb.finish()

if __name__ == "__main__":
    main() 