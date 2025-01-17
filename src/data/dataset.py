from pathlib import Path
import os
import requests
import tarfile
import torch
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from datasets import Dataset
import logging
from tqdm import tqdm

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
        item = {k: v[idx] for k, v in self.encoder_input.items()}
        if self.labels is not None:
            item.update({k: v[idx] for k, v in self.labels.items()})
        return item

class DataProcessor:
    """데이터 처리 및 데이터셋 생성 클래스"""
    
    def __init__(self, tokenizer, config, data_path: str = None):
        self.tokenizer = tokenizer
        self.config = config
        self.data_path = Path(data_path) if data_path else Path(config.general.data_path)
        
    def load_data(self, data_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """데이터 로드"""
        data_path = Path(data_path or self.data_path)
        train_df = pd.read_csv(data_path / 'train.csv')
        val_df = pd.read_csv(data_path / 'dev.csv')
        test_df = pd.read_csv(data_path / 'test.csv')
        
        logging.info(f"Train samples: {len(train_df)}")
        logging.info(f"Validation samples: {len(val_df)}")
        logging.info(f"Test samples: {len(test_df)}")
        
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
        
        if self.config.model.family == "llama":
            return self._prepare_llama_dataset(df, is_train)
        else:
            return self._prepare_bart_dataset(df, is_train)
            
    def _prepare_llama_dataset(self, df: pd.DataFrame, is_train: bool) -> DialogueDataset:
        """Llama 모델용 데이터셋 준비"""
        processed_data = []
        for _, row in df.iterrows():
            inputs = self.tokenizer.model.prepare_inputs_for_generation(
                row['dialogue'],
                row.get('summary', None) if is_train else None
            )
            processed_data.append({
                'input_ids': inputs['input_ids'][0],
                'attention_mask': inputs['attention_mask'][0],
                'labels': inputs.get('labels', None)[0] if 'labels' in inputs else None
            })
            
        return DialogueDataset(
            encoder_input={
                'input_ids': torch.stack([d['input_ids'] for d in processed_data]),
                'attention_mask': torch.stack([d['attention_mask'] for d in processed_data])
            },
            labels={'input_ids': torch.stack([d['labels'] for d in processed_data])} if is_train else None,
            tokenizer=self.tokenizer
        )
        
    def _prepare_bart_dataset(self, df: pd.DataFrame, is_train: bool) -> DialogueDataset:
        """Bart 모델용 데이터셋 준비"""
        encoder_input = self.tokenizer(
            df['dialogue'].tolist(),
            max_length=self.config.tokenizer.encoder_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = None
        if is_train:
            labels = self.tokenizer(
                df['summary'].tolist(),
                max_length=self.config.tokenizer.decoder_max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
        return DialogueDataset(
            encoder_input=encoder_input,
            labels=labels,
            tokenizer=self.tokenizer
        ) 

def load_dataset(data_path: str, split: str):
    """데이터셋 로드"""
    processor = DataProcessor(tokenizer=None, config=None, data_path=data_path)  # data_path 직접 전달
    train_df, val_df, test_df = processor.load_data()
    return train_df if split == "train" else val_df if split == "dev" else test_df 