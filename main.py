import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 토크나이저 병렬화 경고 비활성화

import hydra
from omegaconf import DictConfig
import warnings
import torchvision
from typing import List
import torch
import wandb
from pathlib import Path
from transformers import Seq2SeqTrainingArguments
from omegaconf import OmegaConf
import pandas as pd

# Beta transforms 경고 끄기
torchvision.disable_beta_transforms_warning()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*beta.*")

from src.data.dataset import DataProcessor, download_and_extract, load_dataset
from src.models.model_factory import ModelFactory
from src.utils.utils import save_predictions, get_model_size, print_samples, setup_seeds, init_wandb
from src.utils.metrics import Metrics, TrainerMetrics
from src.trainer import CustomTrainer, WandBCallback
from src.inference import DialogueInference
from src.utils.few_shot_selector import FewShotSelector
from src.data.data_preprocessor import DataPreprocessor

@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    try:
        # config를 기본 Python 타입으로 변환
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # 모든 random seed 설정 (원본 cfg 사용)
        setup_seeds(cfg.general.seed)
        
        # output_dir 초기화 (변환된 cfg_dict 전달)
        output_dir = init_wandb(cfg_dict)
        
        # Ensure data directory exists (원본 cfg 사용)
        download_and_extract(cfg.url.data, cfg.general.data_path)
        
        print("모델 생성 시작...")
        model = ModelFactory.create_model(cfg)  # 원본 cfg 사용
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
            
            # Few-shot 예제 선택 로직
            sample_dialogue = None
            sample_summary = None
            if cfg.custom_config.few_shot:
                examples = FewShotSelector.select_examples(train_df, cfg)
                sample_dialogue = examples["dialogue"]
                sample_summary = examples["summary"]
            else:
                # few-shot이 아닌 경우 기본 예제 사용
                sample_dialogue = train_df.iloc[0]['dialogue']
                sample_summary = train_df.iloc[0]['summary']
            
            # Initialize metrics
            metrics = Metrics(config=cfg)
            
            print(f"\nTesting prompt version {cfg.prompt.version}")
            # 검증 데이터 요약 생성
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
            # 데이터 전처리
            preprocessor = DataPreprocessor(cfg)
            data_dir = preprocessor.prepare_data()  # 전처리된 데이터 또는 원본 데이터 경로 반환
            
            # 데이터셋 준비
            processor = DataProcessor(
                tokenizer=model.tokenizer,
                config=cfg.model,
                data_path=data_dir  # 전처리된 데이터 경로 사용
            )
            
            train_dataset = processor.prepare_dataset("train")
            val_dataset = processor.prepare_dataset("validation" if cfg.train.preprocessing.enabled else "dev")
            
            if cfg.debug.enabled:
                cfg.train.training.num_train_epochs = 1
                train_dataset = torch.utils.data.Subset(
                    train_dataset,
                    range(min(len(train_dataset), cfg.debug.train_samples))
                )
                val_dataset = torch.utils.data.Subset(
                    val_dataset,
                    range(min(len(val_dataset), cfg.debug.val_samples))
                )
            
            # train_config를 config.train.training에서 가져오도록 수정
            train_config = OmegaConf.to_container(cfg.train.training)
            
            # 설정 확인을 위한 디버그 출력
            print("\n=== Training Config ===")
            for k, v in train_config.items():
                print(f"{k}: {v}")
            print("=====================\n")
            
            # Training arguments 설정
            training_args = Seq2SeqTrainingArguments(
                output_dir=str(output_dir),
                run_name=f"{cfg.model.name.split('/')[-1]}_{cfg.model.mode}_{cfg.general.timestamp}",
                
                # 학습 설정
                learning_rate=train_config['learning_rate'],
                num_train_epochs=train_config['num_train_epochs'],
                warmup_ratio=train_config['warmup_ratio'],
                weight_decay=train_config['weight_decay'],
                max_grad_norm=train_config['max_grad_norm'],
                
                # 스케줄러 설정 추가
                lr_scheduler_type=train_config['lr_scheduler_type'],
                warmup_steps=train_config.get('warmup_steps', None),  # warmup_steps가 있으면 사용
                
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
                metric_for_best_model="rouge1_f1",  # eval_ 접두사 제거
                greater_is_better=train_config['greater_is_better'],
                
                # 생성 설정
                predict_with_generate=train_config['predict_with_generate'],
                generation_max_length=cfg.model.generation.max_new_tokens,  # max_new_tokens 대신 generation_max_length 사용
                generation_num_beams=cfg.model.generation.num_beams,
                
                # 로깅
                logging_dir=str(output_dir / "logs"),
                logging_strategy="steps",
                logging_steps=train_config['logging_steps'],
                logging_first_step=train_config['logging_first_step'],
                
                # wandb 관련 설정
                report_to=["wandb"],  # "none" -> ["wandb"]로 변경
                
                # accelerator 관련 설정
                ddp_find_unused_parameters=False,
                no_cuda=not torch.cuda.is_available()
            )
            
            # 메트릭 계산기 초기화
            metrics = TrainerMetrics(
                config=cfg,
                tokenizer=model.tokenizer,
                remove_tokens=cfg.inference.remove_tokens
            )
            
            # CustomTrainer 초기화
            trainer = CustomTrainer(
                model=model.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=model.tokenizer,  # tokenizer 대신 processing_class 사용
                compute_metrics=metrics,
                # remove_tokens 제거 (TrainerMetrics에서 처리)
                callbacks=[WandBCallback()]  # WandB 콜백 명시적 추가
            )
            
            trainer.train()
            
            # 학습 완료 후 샘플 생성
            if not cfg.debug.enabled:
                print("\n=== Generating Sample Prediction ===")
                try:
                    # 입력 준비
                    sample_input = val_dataset[0]['input_ids'].unsqueeze(0).to(model.device)
                    
                    # 생성
                    sample_output = model.model.generate(
                        sample_input,
                        max_new_tokens=cfg.model.generation.max_new_tokens,
                        num_beams=cfg.model.generation.num_beams,
                        length_penalty=cfg.model.generation.length_penalty,
                        repetition_penalty=cfg.model.generation.repetition_penalty,
                        no_repeat_ngram_size=cfg.model.generation.no_repeat_ngram_size,
                        early_stopping=cfg.model.generation.early_stopping,
                        do_sample=cfg.model.generation.do_sample,
                        temperature=cfg.model.generation.temperature,
                        top_p=cfg.model.generation.top_p,
                        top_k=cfg.model.generation.top_k
                    )
                    
                    # 입력 디코딩
                    try:
                        input_ids = sample_input[0].cpu().numpy().tolist()  # numpy 배열을 리스트로 변환
                        print(f"\nSample Input:\n{model.tokenizer.decode(input_ids, skip_special_tokens=True)}")
                    except Exception as e:
                        print(f"\nWarning: Failed to decode input: {e}")
                    
                    # 출력 디코딩
                    try:
                        output_ids = sample_output[0].cpu().numpy().tolist()  # numpy 배열을 리스트로 변환
                        print(f"\nGenerated Summary:\n{model.tokenizer.decode(output_ids, skip_special_tokens=True)}")
                    except Exception as e:
                        print(f"\nWarning: Failed to decode output: {e}")
                    
                    # 레이블 디코딩
                    try:
                        labels = val_dataset[0]['labels'].cpu().numpy()
                        valid_labels = labels[labels != -100].tolist()  # -100 값 제외하고 리스트로 변환
                        if valid_labels:  # 유효한 레이블이 있는 경우에만 디코딩
                            print(f"\nGold Summary:\n{model.tokenizer.decode(valid_labels, skip_special_tokens=True)}")
                        else:
                            print("\nWarning: No valid labels found")
                    except Exception as e:
                        print(f"\nWarning: Failed to decode labels: {e}")
                        print("Labels shape:", labels.shape if hasattr(labels, 'shape') else 'unknown')
                        print("Valid labels:", valid_labels if 'valid_labels' in locals() else 'not created')
                    
                    print("\n" + "="*100)
                    
                except Exception as e:
                    print(f"\nError generating sample: {e}")
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        raise
        
    finally:
        if not cfg.debug.enabled and wandb.run is not None and wandb.run.status != "failed":
            inference = DialogueInference(cfg)
            inference.inference(cfg.general.data_path)
        
        # wandb 정리
        if wandb.run is not None:
            if wandb.run.status != "failed":
                wandb.finish()
            else:
                wandb.finish(exit_code=1)  # 실패 상태로 종료

if __name__ == "__main__":
    main() 