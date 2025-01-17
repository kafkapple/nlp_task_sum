import hydra
from omegaconf import DictConfig
import warnings
import torchvision
from typing import List
import pandas as pd
import os
import numpy as np

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
    
    # OmegaConf를 일반 dict로 변환 (직렬화 가능하도록)
    config_dict = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    
    wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        config=config_dict,  # 변환된 dict 사용
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

@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    # wandb 초기화 및 출력 디렉토리 설정
    output_dir = init_wandb(cfg)
    pl.seed_everything(cfg.general.seed)
    
    # Ensure data directory exists
    download_and_extract(cfg.url.data, cfg.general.data_path)
    
    # 모델 생성
    model = ModelFactory.create_model(cfg)
    
    # 모델 파라미터 수 로깅 (선택적)
    if hasattr(model, 'model'):
        wandb.config.update({
            "model/num_parameters": get_model_size(model.model)
        })
    
    # 프롬프트 모드일 때 few-shot 샘플 준비
    sample_dialogue = None
    sample_summary = None
    
    # 모델 타입에 따른 처리
    if cfg.model.mode == "prompt":
        processor = DataProcessor(None, cfg)
        train_df, val_df, test_df = processor.load_data()
        
        # few-shot 샘플 준비
        if cfg.prompt.mode == "few-shot":
            few_shot_samples = train_df.sample(n=cfg.prompt.n_samples, random_state=cfg.general.seed)
            sample_dialogue = few_shot_samples.iloc[0]['dialogue']
            sample_summary = few_shot_samples.iloc[0]['summary']
            print(f"\nFew-shot 샘플:")
            print(f"대화: {sample_dialogue}")
            print(f"요약: {sample_summary}\n")
        
        # Initialize metrics
        metrics = Metrics()
        
        print(f"\nTesting prompt version {cfg.prompt.version}")
        # 검증 데이터 요약 생성 (한 번만 실행)
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
            test_predictions = model.batch_summarize(
                test_df['dialogue'].tolist(),
                sample_dialogue,
                sample_summary,
                prompt_version=f"v{cfg.prompt.version}"
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
        
    else:  # finetune
        processor = DataProcessor(model.tokenizer, cfg)
        train_dataset = processor.prepare_dataset("train")
        val_dataset = processor.prepare_dataset("dev")
        
        # 데이터셋 유효성 검사 추가
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty!")
            
        # 데이터셋 형식 확인
        sample = train_dataset[0]
        required_keys = ['input_ids', 'attention_mask', 'labels']
        missing_keys = [key for key in required_keys if key not in sample]
        if missing_keys:
            raise ValueError(f"Dataset is missing required keys: {missing_keys}")
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=cfg.general.output_path,
            num_train_epochs=cfg.train.training.num_epochs,
            learning_rate=cfg.train.training.learning_rate,
            per_device_train_batch_size=cfg.train.training.train_batch_size,
            per_device_eval_batch_size=cfg.train.training.eval_batch_size,
            warmup_ratio=cfg.train.training.warmup_ratio,
            weight_decay=cfg.train.training.weight_decay,
            fp16=cfg.train.training.fp16,
            bf16=cfg.train.training.bf16,
            gradient_checkpointing=cfg.train.training.optimization.gradient_checkpointing,
            gradient_accumulation_steps=cfg.train.training.optimization.gradient_accumulation_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            predict_with_generate=True,
        )
        
        # Initialize trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=model.tokenizer,
            compute_metrics=lambda pred: compute_metrics(
                tokenizer=model.tokenizer,
                pred=pred,
                config=cfg
            ),
        )

        # Train model
        trainer.train()
        # Inference
        from src.inference import DialogueInference
        inferencer = DialogueInference(cfg)
        test_file_path = f"{cfg.general.data_path}/test.csv"
        
        # validation set으로 예측 수행
        val_predictions = trainer.predict(val_dataset)
        val_df = pd.read_csv(f"{cfg.general.data_path}/dev.csv")
        
        # 예측 결과 디코딩
        pred_summaries = [
            model.tokenizer.decode(p, skip_special_tokens=True) 
            for p in val_predictions.predictions
        ]
        
        # 샘플 출력
        print_samples(
            val_df['dialogue'].tolist(),
            val_df['summary'].tolist(),
            pred_summaries
        )
        
        # test set 추론
        results = inferencer.inference(test_file_path)
    # Load data
    
    wandb.finish()
    

if __name__ == "__main__":
    main() 