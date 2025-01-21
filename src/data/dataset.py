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
    
    def __init__(self, tokenizer=None, config=None, data_path=None):
        self.tokenizer = tokenizer
        self.config = config
        self.data_path = data_path
        
    def prepare_dataset(self, split: str):
        """데이터셋 준비"""
        # 데이터 로드
        df = pd.read_csv(f"{self.data_path}/{split}.csv")
        
        # 빈 summary 체크 및 처리
        empty_summaries = df['summary'].isna().sum()
        if empty_summaries > 0:
            print(f"Warning: Found {empty_summaries} empty summaries in {split} set")
            df = df.dropna(subset=['summary'])
        
        # 데이터 검증
        print(f"\n=== {split} Dataset Info ===")
        print(f"Total samples: {len(df)}")
        print("\nSample lengths:")
        print(f"Dialogue min/max/mean length: {df['dialogue'].str.len().min()}/{df['dialogue'].str.len().max()}/{df['dialogue'].str.len().mean():.1f}")
        print(f"Summary min/max/mean length: {df['summary'].str.len().min()}/{df['summary'].str.len().max()}/{df['summary'].str.len().mean():.1f}")
        
        # BART 모델용 데이터셋 준비
        return self._prepare_bart_dataset(df, split == "train")
        
    def _prepare_bart_dataset(self, df: pd.DataFrame, is_train: bool) -> DialogueDataset:
        """Bart 모델용 데이터셋 준비"""
        # 1. 입력 데이터 준비
        encoder_input = self.tokenizer(
            df['dialogue'].tolist(),
            max_length=self.config.encoder_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # attention_mask가 3차원이면 2차원으로 변환
        if len(encoder_input['attention_mask'].shape) > 2:
            encoder_input['attention_mask'] = encoder_input['attention_mask'].squeeze(0)
        
        if is_train:
            # 2. 레이블 데이터 준비
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
            
            # 디버깅 출력
            print("\n=== Dataset Debug Info ===")
            print(f"Input shape: {encoder_input['input_ids'].shape}")
            print(f"Attention mask shape: {encoder_input['attention_mask'].shape}")
            print(f"Labels shape: {labels.shape}")
            
            # 샘플 출력
            print("\n=== Sample Data ===")
            for i in range(min(2, len(df))):
                print(f"\n[Sample {i+1}]")
                print(f"Dialogue: {df['dialogue'].iloc[i][:100]}...")
                print(f"Summary: {df['summary'].iloc[i][:100]}...")
                print(f"Input IDs: {encoder_input['input_ids'][i][:50]}...")
                print(f"Labels: {labels[i][:50]}...")
            
            return DialogueDataset(
                encoder_input=encoder_input,
                labels={'labels': labels},
                tokenizer=self.tokenizer
            )
        else:
            return DialogueDataset(
                encoder_input=encoder_input,
                labels=None,
                tokenizer=self.tokenizer
            )

def load_dataset(data_path: str, split: str):
    """데이터셋 로드"""
    processor = DataProcessor(tokenizer=None, config=None, data_path=data_path)  # data_path 직접 전달
    train_df, val_df, test_df = processor.load_data()
    return train_df if split == "train" else val_df if split == "dev" else test_df 