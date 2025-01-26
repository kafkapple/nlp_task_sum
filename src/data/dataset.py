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
from torch.utils.data import Dataset

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


class DialogueDataset(Dataset):
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
    
    def __init__(self, tokenizer=None, config=None, data_path=None):
        self.tokenizer = tokenizer
        self.config = config
        self.data_path = Path(data_path) if data_path else None
        
        # 모델 설정 직접 참조
        self.model_family = config.family
        self.stats = {}
        self.length_ratio = config.generation.get('length_ratio', 0.5)
    
    def _compute_length_stats(self, texts, key_prefix):
        """텍스트 길이 통계 계산 (문자 및 토큰)"""
        char_lengths = [len(text) for text in texts]
        stats = {
            f"{key_prefix}_char_len": {
                "min": min(char_lengths),
                "max": max(char_lengths),
                "mean": sum(char_lengths) / len(char_lengths),
                "p95": np.percentile(char_lengths, 95)
            }
        }
        
        if self.tokenizer:
            tokens = self.tokenizer(
                texts, 
                truncation=True,
                max_length=self.config.tokenizer.max_length,
                padding=False
            )
            token_lengths = [len(ids) for ids in tokens['input_ids']]
            stats[f"{key_prefix}_token_len"] = {
                "min": min(token_lengths),
                "max": max(token_lengths),
                "mean": sum(token_lengths) / len(token_lengths),
                "p95": np.percentile(token_lengths, 95)
            }
        
        return stats
    
    def _adjust_generation_length(self, input_length):
        """입력 길이에 따른 생성 길이 동적 조절"""
        base_length = int(input_length * self.length_ratio)
        
        # 최소/최대 길이 제한 적용
        min_tokens = self.config.generation.min_new_tokens
        max_tokens = self.config.generation.max_new_tokens
        
        return max(min_tokens, min(base_length, max_tokens))
    
    def _prepare_t5_dataset(self, df, is_train=True):
        """T5 모델용 데이터셋 준비"""
        if is_train and 'summary' not in df.columns:
            raise ValueError("학습 데이터셋에 'summary' 컬럼이 필요합니다.")
        
        # 입력 텍스트 토크나이징
        inputs = self.tokenizer(
            df['dialogue'].tolist(),
            padding=True,
            truncation=True,
            max_length=self.config.tokenizer.max_length,
            return_tensors="pt"
        )
        
        # 학습 모드인 경우 레이블도 토크나이징
        if is_train:
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    df['summary'].tolist(),
                    padding=True,
                    truncation=True,
                    max_length=self.config.tokenizer.max_target_length,
                    return_tensors="pt"
                )
            
            # 패딩 토큰을 -100으로 변경 (loss 계산에서 제외)
            labels['input_ids'][labels['input_ids'] == self.tokenizer.pad_token_id] = -100
            
            return DialogueDataset(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=labels['input_ids']
            )
        
        # 추론 모드
        return DialogueDataset(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
    
    def _prepare_bart_dataset(self, df, is_train=True):
        """BART 모델용 데이터셋 준비"""
        if is_train and 'summary' not in df.columns:
            raise ValueError("학습 데이터셋에 'summary' 컬럼이 필요합니다.")
        
        # 입력 텍스트 토크나이징
        inputs = self.tokenizer(
            df['dialogue'].tolist(),
            padding=True,
            truncation=True,
            max_length=self.config.tokenizer.max_length,
            return_tensors="pt"
        )
        
        # 학습 모드인 경우 레이블도 토크나이징
        if is_train:
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    df['summary'].tolist(),
                    padding=True,
                    truncation=True,
                    max_length=self.config.tokenizer.max_target_length,
                    return_tensors="pt"
                )
            
            # 패딩 토큰을 -100으로 변경 (loss 계산에서 제외)
            labels['input_ids'][labels['input_ids'] == self.tokenizer.pad_token_id] = -100
            
            return DialogueDataset(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=labels['input_ids']
            )
        
        # 추론 모드
        return DialogueDataset(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
    
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
            max_length=self.config.tokenizer.max_length,
            return_tensors="pt"
        )
        
        # 레이블 생성
        labels = tokenized['input_ids'].clone()
        
        # dialogue 부분을 -100으로 마스킹
        for i, prompt in enumerate(prompts):
            summary_start = prompt.find("### Summary:\n") + len("### Summary:\n")
            summary_tokens = self.tokenizer(prompt[summary_start:], 
                                          truncation=True,
                                          max_length=self.config.tokenizer.max_length).input_ids
            if len(summary_tokens) > 1:  # bos/eos 토큰 제외
                # dialogue 부분만 -100으로 마스킹
                dialogue_end = len(tokenized['input_ids'][i]) - len(summary_tokens)
                labels[i, :dialogue_end] = -100
        
        # 데이터셋 반환
        return DialogueDataset(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
            labels=labels
        )
    
    def prepare_dataset(self, split: str):
        """데이터셋 준비"""
        df = pd.read_csv(self.data_path / f"{split}.csv")
        
        # 길이 통계 계산
        dialogue_stats = self._compute_length_stats(
            df['dialogue'].tolist(), 'dialogue'
        )
        self.stats[f"{split}_dialogue"] = dialogue_stats
        
        if 'summary' in df.columns:
            summary_stats = self._compute_length_stats(
                df['summary'].tolist(), 'summary'
            )
            self.stats[f"{split}_summary"] = summary_stats
        
        # 통계 저장
        output_path = getattr(self.config, 'output_path', 'outputs')
        stats_path = Path(output_path) / "data_stats"
        stats_path.mkdir(parents=True, exist_ok=True)
        with open(stats_path / f"{split}_stats.json", 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # 동적 생성 길이 설정
        if self.tokenizer and 'dialogue' in df.columns:
            dialogue_stats = self.stats[f"{split}_dialogue"]
            token_stats = dialogue_stats.get('dialogue_token_len', {})
            if token_stats:
                avg_input_length = int(token_stats['mean'])
                new_max_tokens = self._adjust_generation_length(
                    avg_input_length
                )
                
                # 모델 설정 업데이트
                self.config.generation.max_new_tokens = new_max_tokens
        
        # 모델별 데이터셋 준비
        is_train = 'summary' in df.columns
        if self.model_family == 't5':
            return self._prepare_t5_dataset(df, is_train)
        elif self.model_family == 'bart':
            return self._prepare_bart_dataset(df, is_train)
        elif self.model_family == 'llama':
            return self._prepare_llama_dataset(df)
        else:
            raise ValueError(f"Unknown model family: {self.model_family}")


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