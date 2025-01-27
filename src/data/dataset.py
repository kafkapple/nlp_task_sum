from pathlib import Path
import os
import requests
import tarfile
import torch
import pandas as pd
from typing import Optional, Dict
from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
import json
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset

def download_and_extract(url: str, data_path: str):
    """데이터 다운로드 및 압축 해제"""
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    
    if (data_path / "train.csv").exists():
        print("데이터가 이미 존재합니다.")
        return
        
    print(f"데이터 다운로드 중... URL: {url}")
    zip_path = data_path / "data.tar.gz"
    
    try:
        session = requests.Session()
        session.verify = False
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        # 다운로드 진행률 표시
        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
                
        # 압축 해제 및 파일 정리
        with tarfile.open(zip_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.startswith('data/'):
                    member.name = member.name.replace('data/', '', 1)
            tar.extractall(path=data_path)
            
        # 중복 폴더 정리
        nested_data_dir = data_path / "data"
        if nested_data_dir.exists() and nested_data_dir.is_dir():
            for file_path in nested_data_dir.iterdir():
                target_path = data_path / file_path.name
                if not target_path.exists():
                    file_path.rename(target_path)
            nested_data_dir.rmdir()
            
    finally:
        if zip_path.exists():
            os.remove(zip_path)


class BaseDataset(ABC):
    """기본 데이터셋 클래스"""
    
    def __init__(self, encoder_input: Dict[str, torch.Tensor], 
                 labels: Optional[Dict[str, torch.Tensor]] = None):
        self.encoder_input = encoder_input
        self.labels = labels
        
    def __len__(self):
        return len(self.encoder_input['input_ids'])
        
    @abstractmethod
    def __getitem__(self, idx):
        pass


class DialogueDataset(TorchDataset):
    """통합 대화 요약 데이터셋"""
    def __init__(self, input_ids, attention_mask, labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item


class T5Dataset(BaseDataset):
    """T5 모델용 데이터셋"""
    
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encoder_input.items()}
        if self.labels is not None:
            item['labels'] = self.labels['labels'][idx]
            # T5 specific processing if needed
        return item


class DataProcessor:
    """데이터 처리 및 데이터셋 생성 클래스"""
    
    def __init__(self, tokenizer, config, data_path):
        self.tokenizer = tokenizer
        self.config = config
        self.data_path = data_path
        
    def prepare_dataset(self, split: str, use_prompt: bool = False):
        """데이터셋 준비
        
        Args:
            split: 데이터셋 분할("train", "validation", "test")
            use_prompt: 프롬프트 템플릿 사용 여부
        """
        # 데이터 로드
        train_df, val_df, test_df = load_dataset(self.data_path)
        
        if split == "train":
            df = train_df
        elif split in ["validation", "dev"]:
            df = val_df
        else:
            df = test_df
            
        # Few-shot 예제 선택 (학습 시에만)
        sample_dialogue = None
        sample_summary = None
        if use_prompt and hasattr(self.config, 'custom_config') and self.config.custom_config.few_shot:
            from src.utils.few_shot_selector import FewShotSelector
            examples = FewShotSelector.select_examples(train_df, self.config)
            sample_dialogue = examples["dialogue"]
            sample_summary = examples["summary"]
            
        def preprocess_function(examples):
            if use_prompt:
                # 프롬프트 템플릿 적용
                version = self.config.prompt.version.replace('v', '') if self.config.prompt.version else "1"
                template = self.config.prompt_templates[version]
                
                if sample_dialogue and sample_summary:
                    # Few-shot 프롬프트
                    if version == "1":
                        prompts = [
                            template["few_shot"].format(
                                sample_dialogue=sample_dialogue,
                                sample_summary=sample_summary,
                                dialogue=d
                            ) for d in examples["dialogue"]
                        ]
                    else:  # version == "2"
                        prompts = [
                            template["system"] + "\n" +
                            template["instruction"] + "\n" +
                            template["few_shot"]["user"].format(
                                sample_dialogue=sample_dialogue
                            ) + "\n" +
                            template["few_shot"]["assistant"].format(
                                sample_summary=sample_summary
                            ) + "\n" +
                            template["few_shot"]["final_user"].format(
                                dialogue=d
                            ) for d in examples["dialogue"]
                        ]
                else:
                    # Zero-shot 프롬프트
                    if version == "1":
                        prompts = [
                            template["zero_shot"].format(dialogue=d)
                            for d in examples["dialogue"]
                        ]
                    else:  # version == "2"
                        prompts = [
                            template["system"] + "\n" +
                            template["instruction"] + "\n" +
                            template["zero_shot"].format(dialogue=d)
                            for d in examples["dialogue"]
                        ]
                
                # 토크나이징
                model_inputs = self.tokenizer(
                    prompts,
                    max_length=self.config.tokenizer.encoder_max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
            else:
                # 기존 방식 (프롬프트 없이)
                model_inputs = self.tokenizer(
                    examples["dialogue"],
                    max_length=self.config.tokenizer.encoder_max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
            
            # 레이블 토크나이징
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples["summary"],
                    max_length=self.config.tokenizer.decoder_max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
            
        # 데이터셋 생성
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # 첫 번째 예제의 프롬프트 출력 (디버깅용)
        if use_prompt and len(dataset) > 0:
            print("\n=== Training Prompt Example ===")
            decoded_prompt = self.tokenizer.decode(dataset[0]['input_ids'], skip_special_tokens=True)
            print(decoded_prompt)
            print("=" * 50)
        
        return dataset


def load_dataset(data_path: str, split: Optional[str] = None) -> pd.DataFrame:
    """CSV 파일에서 데이터셋 직접 로드"""
    data_path = Path(data_path)
    processed_path = data_path / "processed"
    
    # 전처리된 데이터가 없으면 전처리 수행
    if not processed_path.exists() or not (processed_path / "train.csv").exists():
        from src.data.data_preprocessor import DataPreprocessor
        from hydra import compose, initialize
        from hydra.core.global_hydra import GlobalHydra
        
        # Hydra 초기화 전에 clear
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
            
        # Hydra 초기화 및 config 로드
        with initialize(version_base=None, config_path="../../config"):
            cfg = compose(config_name="config")
            cfg.general.data_path = str(data_path)
            
            # 전처리 설정 활성화
            cfg.train.preprocessing.enabled = True
            cfg.train.preprocessing.merge_train_dev = True
            cfg.train.preprocessing.split_ratio = 0.2
            cfg.train.preprocessing.stratify_by_length = True
            cfg.train.preprocessing.length_strategy = "ratio"
            cfg.train.preprocessing.random_state = cfg.general.seed
            
            # 전처리 수행
            preprocessor = DataPreprocessor(cfg)
            processed_path = preprocessor.prepare_data()
    
    # 파일 경로 설정 (processed_path가 이미 processed 디렉토리를 가리킴)
    train_path = processed_path / "train.csv"
    dev_path = processed_path / "dev.csv"
    test_path = processed_path / "test.csv"
    
    if split is None:
        # 모든 split 반환
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(dev_path)
        test_df = pd.read_csv(test_path)
        return train_df, val_df, test_df
    else:
        # 특정 split만 반환
        if split in ["validation", "dev"]:
            return pd.read_csv(dev_path)
        elif split == "test":
            return pd.read_csv(test_path)
        else:  # train
            return pd.read_csv(train_path) 