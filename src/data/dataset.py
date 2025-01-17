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
        
        # labels는 항상 포함 (없으면 -100으로 채움)
        if self.labels is not None:
            item.update({k: v[idx] for k, v in self.labels.items()})
        else:
            item['labels'] = torch.full_like(item['input_ids'], -100)
            
        return item

class DataProcessor:
    """데이터 처리 및 데이터셋 생성 클래스"""
    
    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.config = cfg
        self.data_path = Path(cfg.general.data_path)
        # 초기화 시 seed 설정
        if hasattr(cfg.general, 'seed'):
            random.seed(cfg.general.seed)
            np.random.seed(cfg.general.seed)
            
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
        
        if self.config.model.family == "llama":
            return self._prepare_llama_dataset(df, is_train)
        else:
            return self._prepare_bart_dataset(df, is_train)
            
    def _prepare_llama_dataset(self, df: pd.DataFrame, is_train: bool) -> HFDataset:
        """Llama 모델용 데이터셋 준비"""
        processed_data = []
        for _, row in df.iterrows():
            # 프롬프트 생성
            if is_train:
                prompt = f"Dialogue:\n{row['dialogue']}\nSummary:\n{row['summary']}"
            else:
                prompt = f"Dialogue:\n{row['dialogue']}\nSummary:"
            
            # 토크나이징
            inputs = self.tokenizer(
                prompt,
                max_length=self.config.model.tokenizer.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            if is_train:
                labels = inputs['input_ids'].clone()
                
                # "Summary:" 토큰의 위치를 더 안전하게 찾기
                try:
                    # 전체 텍스트에서 "Summary:" 위치 찾기
                    summary_text_pos = prompt.find("Summary:")
                    if summary_text_pos != -1:
                        # Summary: 이전까지의 텍스트를 토크나이징
                        prefix_tokens = self.tokenizer(
                            prompt[:summary_text_pos],
                            add_special_tokens=False
                        )['input_ids']
                        summary_start = len(prefix_tokens)
                        
                        # Summary: 이전 부분을 -100으로 마스킹
                        labels[0, :summary_start] = -100
                    else:
                        # Summary: 를 찾지 못한 경우 전체를 마스킹
                        labels[0, :] = -100
                except:
                    # 오류 발생 시 전체를 마스킹
                    labels[0, :] = -100
            else:
                labels = torch.full_like(inputs['input_ids'], -100)
            
            processed_data.append({
                'input_ids': inputs['input_ids'][0].tolist(),
                'attention_mask': inputs['attention_mask'][0].tolist(),
                'labels': labels[0].tolist()
            })
        
        # HuggingFace Dataset 생성
        return HFDataset.from_dict({
            'input_ids': [d['input_ids'] for d in processed_data],
            'attention_mask': [d['attention_mask'] for d in processed_data],
            'labels': [d['labels'] for d in processed_data]
        })
        
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