from pathlib import Path
import os
import requests
import tarfile
import torch
import pandas as pd
from typing import Optional, Dict
from tqdm import tqdm
from abc import ABC, abstractmethod

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


class DialogueDataset(BaseDataset):
    """대화 요약 데이터셋 클래스"""
    
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encoder_input.items()}
        if self.labels is not None:
            item['labels'] = self.labels['labels'][idx]
        else:
            item['labels'] = torch.full_like(item['input_ids'], -100)
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
    
    def __init__(self, tokenizer=None, config=None, data_path=None):
        self.tokenizer = tokenizer
        self.config = config
        self.data_path = Path(data_path) if data_path else None
        self.model_type = config.model_type if config and hasattr(config, 'model_type') else 'bart'
    
    def load_data(self):
        """CSV 파일들을 데이터프레임으로 로드"""
        if not self.data_path:
            raise ValueError("data_path must be specified")
            
        required_files = ["train.csv", "dev.csv", "test.csv"]
        for file in required_files:
            if not (self.data_path / file).exists():
                raise ValueError(f"Required file {file} not found in {self.data_path}")
        
        data = {
            'train': pd.read_csv(self.data_path / "train.csv"),
            'dev': pd.read_csv(self.data_path / "dev.csv"),
            'test': pd.read_csv(self.data_path / "test.csv")
        }
        
        print("\n=== Dataset Statistics ===")
        for split, df in data.items():
            print(f"{split.capitalize()} set size: {len(df)}")
        
        return data['train'], data['dev'], data['test']
    
    def prepare_dataset(self, split: str):
        """데이터셋 준비"""
        df = pd.read_csv(self.data_path / f"{split}.csv")
        self._validate_data(df, split)
        
        # 모델 타입에 따른 데이터셋 준비
        is_train = 'summary' in df.columns
        if self.model_type == 't5':
            return self._prepare_t5_dataset(df, is_train)
        else:  # default to bart
            return self._prepare_bart_dataset(df, is_train)
    
    def _validate_data(self, df: pd.DataFrame, split: str):
        """데이터 유효성 검증"""
        print(f"\n=== {split} Dataset Info ===")
        print(f"Total samples: {len(df)}")
        
        # 길이 통계
        dialogue_lengths = df['dialogue'].str.len()
        print("\nDialogue lengths:")
        print(f"min/max/mean: {dialogue_lengths.min()}/{dialogue_lengths.max()}/{dialogue_lengths.mean():.1f}")
        
        if 'summary' in df.columns:
            empty_summaries = df['summary'].isna().sum()
            if empty_summaries > 0:
                print(f"Warning: Found {empty_summaries} empty summaries")
                df.dropna(subset=['summary'], inplace=True)
            
            summary_lengths = df['summary'].str.len()
            print("\nSummary lengths:")
            print(f"min/max/mean: {summary_lengths.min()}/{summary_lengths.max()}/{summary_lengths.mean():.1f}")
    
    def _prepare_bart_dataset(self, df: pd.DataFrame, is_train: bool) -> DialogueDataset:
        """BART 모델용 데이터셋 준비"""
        encoder_input = self._tokenize_text(
            df['dialogue'].tolist(),
            max_length=self.config.encoder_max_len
        )
        
        if not is_train:
            return DialogueDataset(encoder_input=encoder_input)
        
        labels = self._tokenize_text(
            df['summary'].tolist(),
            max_length=self.config.decoder_max_len
        )
        labels = self._prepare_labels(labels['input_ids'])
        
        return DialogueDataset(
            encoder_input=encoder_input,
            labels={'labels': labels}
        )
    
    def _prepare_t5_dataset(self, df: pd.DataFrame, is_train: bool) -> T5Dataset:
        """T5 모델용 데이터셋 준비"""
        # T5 specific preprocessing if needed
        encoder_input = self._tokenize_text(
            df['dialogue'].tolist(),
            max_length=self.config.encoder_max_len
        )
        
        if not is_train:
            return T5Dataset(encoder_input=encoder_input)
        
        labels = self._tokenize_text(
            df['summary'].tolist(),
            max_length=self.config.decoder_max_len
        )
        labels = self._prepare_labels(labels['input_ids'])
        
        return T5Dataset(
            encoder_input=encoder_input,
            labels={'labels': labels}
        )
    
    def _tokenize_text(self, texts: list, max_length: int) -> Dict[str, torch.Tensor]:
        """텍스트 토큰화"""
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    def _prepare_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """레이블 준비 (패딩 토큰을 -100으로 변환)"""
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels


def load_dataset(data_path: str, split: str = None):
    """데이터셋 로드 유틸리티 함수"""
    processor = DataProcessor(data_path=data_path)
    train_df, val_df, test_df = processor.load_data()
    
    if split is None:
        return train_df, val_df, test_df
    
    split_map = {'train': train_df, 'dev': val_df, 'test': test_df}
    return split_map.get(split) 