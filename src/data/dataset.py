from pathlib import Path
import os
import requests
import tarfile
import torch
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from datasets import Dataset as HFDataset
import logging
from tqdm import tqdm
import random
import numpy as np

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

class DialogueDataset:
    """대화 요약 데이터셋 클래스"""
    
    def __init__(self, encoder_input: Dict[str, torch.Tensor], 
                 labels: Optional[Dict[str, torch.Tensor]] = None,
                 tokenizer = None):
        self.encoder_input = encoder_input
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.encoder_input['input_ids'])
        
    def __getitem__(self, idx):
        # 기본 입력
        item = {k: v[idx] for k, v in self.encoder_input.items()}
        
        # labels가 있는 경우 추가
        if self.labels is not None:
            item['labels'] = self.labels['labels'][idx]
        else:
            # labels가 없는 경우 -100으로 채움
            item['labels'] = torch.full_like(item['input_ids'], -100)
            
        return item

class DataProcessor:
    """데이터 처리 및 데이터셋 생성 클래스"""
    
    def __init__(self, tokenizer, config, data_path):
        self.tokenizer = tokenizer
        self.config = config  # tokenizer 설정만 포함
        self.data_path = Path(data_path)  # data_path 추가
    
    def load_data(self, data_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """데이터 로드"""
        data_path = Path(data_path or self.data_path)
        train_df = pd.read_csv(data_path / 'train.csv')
        val_df = pd.read_csv(data_path / 'dev.csv')
        test_df = pd.read_csv(data_path / 'test.csv')
        
        logging.info(f"Train samples: {len(train_df)}")
        logging.info(f"Validation samples: {len(val_df)}")
        logging.info(f"Test samples: {len(test_df)}")
        
        # seed 설정 확인
        if hasattr(self.cfg.general, 'seed'):
            np.random.seed(self.cfg.general.seed)
            
        return train_df, val_df, test_df
        
    def get_few_shot_samples(self, train_df: pd.DataFrame, 
                            n_samples: int = 1, 
                            random_seed: int = 2001) -> pd.DataFrame:
        """few-shot 학습용 샘플 추출"""
        return train_df.sample(n_samples, random_state=random_seed)
        
    def prepare_dataset(self, mode: str = "train") -> DialogueDataset:
        """데이터셋 준비"""
        df = pd.read_csv(self.data_path / f"{mode}.csv")
        is_train = mode == "train"
        
        # BART 모델용 데이터셋만 준비 (llama 관련 코드 제거)
        return self._prepare_bart_dataset(df, is_train)
            
    def _prepare_bart_dataset(self, df: pd.DataFrame, is_train: bool) -> DialogueDataset:
        """Bart 모델용 데이터셋 준비"""
        encoder_input = self.tokenizer(
            df['dialogue'].tolist(),
            max_length=self.config.encoder_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        if is_train:
            # 레이블 데이터 준비
            decoder_input = self.tokenizer(
                df['summary'].tolist(),
                max_length=self.config.decoder_max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # labels는 decoder input과 동일하지만, padding token을 -100으로 변경
            labels = decoder_input['input_ids'].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            # labels를 딕셔너리 형태로 전달
            labels_dict = {
                'labels': labels
            }
        else:
            labels_dict = None
            
        return DialogueDataset(
            encoder_input=encoder_input,
            labels=labels_dict,
            tokenizer=self.tokenizer
        )

def load_dataset(data_path: str, split: str):
    """데이터셋 로드"""
    processor = DataProcessor(tokenizer=None, config=None, data_path=data_path)  # data_path 직접 전달
    train_df, val_df, test_df = processor.load_data()
    return train_df if split == "train" else val_df if split == "dev" else test_df 