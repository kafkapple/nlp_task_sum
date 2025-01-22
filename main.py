import hydra
from omegaconf import DictConfig
import warnings
import torchvision
from typing import List
import pandas as pd
import os
import numpy as np
import random
import torch
import wandb
from pathlib import Path
from transformers import Seq2SeqTrainingArguments, GenerationConfig
import pytorch_lightning as pl
from datetime import datetime
from omegaconf import OmegaConf

# Beta transforms 경고 끄기
torchvision.disable_beta_transforms_warning()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*beta.*")

from src.data.dataset import DataProcessor, download_and_extract, load_dataset
from src.models.model_factory import ModelFactory
from src.utils.utils import save_predictions
from src.utils.metrics import Metrics, TrainerMetrics
from src.trainer import CustomTrainer, WandBCallback

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

@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    try:
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
            if wandb.run is not None:
                wandb.config.update({
                    "model/num_parameters": int(get_model_size(model.model))
                }, allow_val_change=True)
        
        if cfg.model.mode == "prompt":
            print("프롬프트 모드로 실행 중...")
            processor = DataProcessor(
                tokenizer=None, 
                config=None,
                data_path=cfg.general.data_path
            )
            print("데이터 로드 중...")
            train_df, val_df, test_df = load_dataset(
                data_path=cfg.general.data_path,
                split=None  # 모든 데이터셋 로드
            )
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
            # DataProcessor 초기화 시 기본값 설정 추가
            tokenizer_config = OmegaConf.create({
                "encoder_max_len": 512,  # 기본값 설정
                "decoder_max_len": 128,  # 기본값 설정
                **cfg.model.tokenizer  # 기존 설정 유지
            })
            
            processor = DataProcessor(
                tokenizer=model.tokenizer,
                config=tokenizer_config,  # 수정된 설정 사용
                data_path=cfg.general.data_path
            )
            train_dataset = processor.prepare_dataset("train")
            val_dataset = processor.prepare_dataset("dev")
            
            # train_config를 config.train.training에서 가져오도록 수정
            train_config = OmegaConf.to_container(cfg.train.training)
            
            # 설정 확인을 위한 디버그 출력
            print("\n=== Training Config ===")
            for k, v in train_config.items():
                print(f"{k}: {v}")
            print("=====================\n")
            
            # generation 설정 가져오기
            generation_config = {
                "max_new_tokens": cfg.model.generation.max_new_tokens,
                "min_new_tokens": cfg.model.generation.min_new_tokens,
                "num_beams": cfg.model.generation.num_beams,
                "temperature": cfg.model.generation.temperature,
                "top_p": cfg.model.generation.top_p,
                "do_sample": cfg.model.generation.do_sample,
                "length_penalty": cfg.model.generation.length_penalty,
                "repetition_penalty": cfg.model.generation.repetition_penalty,
                "no_repeat_ngram_size": cfg.model.generation.no_repeat_ngram_size,
                "early_stopping": cfg.model.generation.early_stopping
            }

            # Training arguments 설정
            training_args = Seq2SeqTrainingArguments(
                output_dir=str(output_dir),
                run_name=f"{cfg.model.name}_{cfg.model.mode}_{cfg.general.timestamp}",
                
                # 학습 설정
                learning_rate=train_config['learning_rate'],
                num_train_epochs=train_config['num_train_epochs'],
                warmup_ratio=train_config['warmup_ratio'],
                weight_decay=train_config['weight_decay'],
                max_grad_norm=train_config['max_grad_norm'],
                
                # 배치 및 최적화 설정
                per_device_train_batch_size=train_config['per_device_train_batch_size'],
                per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
                gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
                gradient_checkpointing=train_config['gradient_checkpointing'],
                
                # 하드웨어 최적화
                fp16=train_config['fp16'],
                bf16=train_config['bf16'],
                optim=train_config['optim'],
                
                # 평가 및 저장 전략
                evaluation_strategy=train_config['evaluation_strategy'],
                save_strategy=train_config['save_strategy'],
                save_total_limit=train_config['save_total_limit'],
                load_best_model_at_end=train_config['load_best_model_at_end'],
                metric_for_best_model=train_config['metric_for_best_model'],
                greater_is_better=train_config['greater_is_better'],
                
                # 생성 설정
                predict_with_generate=train_config['predict_with_generate'],
                generation_max_length=cfg.model.generation.max_new_tokens,
                generation_num_beams=cfg.model.generation.num_beams,
                
                # 로깅
                logging_steps=train_config['logging_steps'],
                logging_first_step=train_config['logging_first_step'],
                
                # wandb 관련 설정
                report_to=[],
                disable_tqdm=False,
                
                # accelerator 관련 설정
                ddp_find_unused_parameters=False,
                no_cuda=not torch.cuda.is_available()
            )
            
            # 메트릭 계산기 초기화
            metrics = TrainerMetrics(
                tokenizer=model.tokenizer,
                remove_tokens=cfg.inference.remove_tokens
            )
            
            # CustomTrainer 초기화
            trainer = CustomTrainer(
                model=model.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=model.tokenizer,
                compute_metrics=metrics,
                remove_tokens=cfg.inference.remove_tokens,
                callbacks=[WandBCallback()]  # WandB 콜백 명시적 추가
            )
            
            trainer.train()
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        if wandb.run is not None:
            wandb.finish()
        raise
        
    finally:
        # 프로그램 종료 시 wandb 정리
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main() 