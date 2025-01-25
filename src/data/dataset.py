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


class DialogueDataset(torch.utils.data.Dataset):
    """통합 대화 요약 데이터셋"""
    def __init__(self, input_ids, attention_mask, labels, decoder_input_ids=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.decoder_input_ids = decoder_input_ids
        
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
        
        # seq2seq 모델을 위한 decoder_input_ids 추가
        if self.decoder_input_ids is not None:
            item['decoder_input_ids'] = self.decoder_input_ids[idx]
            
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
        self.data_path = Path(data_path)
        
        # LLaMA 모델을 위한 특별 처리
        self.is_llama = hasattr(tokenizer, 'name_or_path') and 'llama' in tokenizer.name_or_path.lower()
    
    def prepare_dataset(self, split):
        """데이터셋 준비"""
        # CSV 파일 직접 로드
        df = pd.read_csv(self.data_path / f"{split}.csv")
        
        if self.is_llama:
            return self._prepare_llama_dataset(df)
        else:
            return self._prepare_seq2seq_dataset(df)
    
    def _prepare_llama_dataset(self, df):
        """LLaMA 모델용 데이터셋 준비"""
        # 데이터 유효성 검사
        if 'dialogue' not in df.columns or 'summary' not in df.columns:
            raise ValueError("데이터셋에 'dialogue'와 'summary' 컬럼이 필요합니다.")
        
        # LLaMA 형식의 프롬프트 생성
        prompts = [
            f"### Dialogue:\n{dialogue}\n\n### Summary:\n{summary}"
            for dialogue, summary in zip(df['dialogue'], df['summary'])
        ]
        
        # 토크나이징
        tokenized = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # 레이블 생성
        labels = tokenized['input_ids'].clone()
        
        # dialogue 부분을 -100으로 마스킹
        for i, prompt in enumerate(prompts):
            summary_start = prompt.find("### Summary:\n") + len("### Summary:\n")
            summary_tokens = self.tokenizer(prompt[summary_start:]).input_ids
            if len(summary_tokens) > 1:  # bos/eos 토큰 제외
                labels[i, :-len(summary_tokens)+1] = -100  # 마지막 eos 토큰 유지
        
        # 커스텀 데이터셋 반환
        return DialogueDataset(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
            labels=labels
        )
    
    def _prepare_seq2seq_dataset(self, df):
        """기존 seq2seq 모델용 데이터셋 준비"""
        # 데이터 유효성 검사
        if 'dialogue' not in df.columns or 'summary' not in df.columns:
            raise ValueError("데이터셋에 'dialogue'와 'summary' 컬럼이 필요합니다.")
        
        # 입력 텍스트 토크나이징
        encoder_inputs = self.tokenizer(
            df['dialogue'].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.config.encoder_max_len,
            return_tensors="pt"
        )
        
        # 레이블(요약) 토크나이징
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                df['summary'].tolist(),
                padding="max_length",
                truncation=True,
                max_length=self.config.decoder_max_len,
                return_tensors="pt"
            )
        
        # 패딩 토큰을 -100으로 변경 (loss 계산에서 제외)
        labels['input_ids'][labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        
        # 커스텀 데이터셋 반환
        return DialogueDataset(
            input_ids=encoder_inputs['input_ids'],
            attention_mask=encoder_inputs['attention_mask'],
            labels=labels['input_ids']
        )


def load_dataset(data_path: str, split: Optional[str] = None) -> pd.DataFrame:
    """CSV 파일에서 데이터셋 직접 로드"""
    data_path = Path(data_path)
    
    if split is None:
        # 모든 split 반환
        train_df = pd.read_csv(data_path / "train.csv")
        val_df = pd.read_csv(data_path / "dev.csv")
        test_df = pd.read_csv(data_path / "test.csv")
        return train_df, val_df, test_df
    else:
        # 특정 split만 반환
        filename = "dev.csv" if split == "dev" else f"{split}.csv"
        return pd.read_csv(data_path / filename) 