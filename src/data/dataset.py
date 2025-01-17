from torch.utils.data import Dataset
import torch
from typing import Dict, Optional
import os
import pandas as pd
import requests
import zipfile
from pathlib import Path
from datasets import Dataset

def download_and_extract(url: str, data_path: str):
    """데이터 다운로드 및 압축 해제"""
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # 이미 파일이 존재하는지 확인
    if (data_path / "train.csv").exists():
        print("데이터가 이미 존재합니다.")
        return
        
    # 데이터 다운로드
    print(f"데이터 다운로드 중... URL: {url}")
    zip_path = data_path / "data.tar.gz"  # 확장자 변경
    try:
        # requests 세션 생성 및 설정
        session = requests.Session()
        session.verify = False  # SSL 검증 비활성화
        
        # 스트리밍 방식으로 다운로드
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        # 파일 크기 확인
        total_size = int(response.headers.get('content-length', 0))
        print(f"다운로드 파일 크기: {total_size/1024/1024:.2f} MB")
        
        # 청크 단위로 파일 저장
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        print("다운로드 완료. 압축 해제 중...")
        
        # 파일 크기 확인
        if os.path.getsize(zip_path) == 0:
            raise Exception("다운로드된 파일이 비어있습니다.")
            
        # tar.gz 파일 압축 해제
        import tarfile
        try:
            with tarfile.open(zip_path, 'r:gz') as tar:
                print("TAR 파일 내용:")
                for member in tar.getmembers():
                    print(f" - {member.name}: {member.size:,} bytes")
                tar.extractall(path=data_path)
                print("압축 해제 완료")
        except tarfile.ReadError:
            print("TAR 파일이 아니거나 손상되었습니다.")
            raise
            
        # 임시 파일 삭제
        if zip_path.exists():
            os.remove(zip_path)
            
    except requests.exceptions.RequestException as e:
        print(f"다운로드 실패: {e}")
        if zip_path.exists():
            os.remove(zip_path)
        raise
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
        if zip_path.exists():
            os.remove(zip_path)
        raise

class DataProcessor:
    """데이터 전처리를 위한 클래스"""
    
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.max_source_length = config.model.max_source_length
        self.max_target_length = config.model.max_target_length
        
    def _load_data(self, split: str) -> pd.DataFrame:
        """CSV 파일에서 데이터 로드"""
        data_path = Path(self.config.general.data_path)
        return pd.read_csv(data_path / f"{split}.csv")
        
    def _preprocess_function(self, examples):
        """데이터 전처리 함수"""
        # 입력 텍스트 토크나이징
        model_inputs = self.tokenizer(
            examples["dialogue"],
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
        )
        
        # 타겟 텍스트 토크나이징
        labels = self.tokenizer(
            examples["summary"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    def prepare_dataset(self, split: str) -> Dataset:
        """데이터셋 준비"""
        # 데이터 로드
        df = self._load_data(split)
        
        # Huggingface Dataset으로 변환
        dataset = Dataset.from_pandas(df)
        
        # 전처리 적용
        processed_dataset = dataset.map(
            self._preprocess_function,
            remove_columns=dataset.column_names,
            batch_size=self.config.train.training.train_batch_size,
        )
        
        return processed_dataset

class DialogueDataset(Dataset):
    """대화 요약 데이터셋"""
    
    def __init__(self, 
                 encoder_input: Dict[str, torch.Tensor],
                 labels: Optional[Dict[str, torch.Tensor]] = None,
                 decoder_input: Optional[Dict[str, torch.Tensor]] = None,
                 tokenizer = None):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        
        if self.decoder_input is not None:
            decoder_item = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}
            item['decoder_input_ids'] = decoder_item['input_ids']
            item['decoder_attention_mask'] = decoder_item['attention_mask']
            
        if self.labels is not None:
            labels = self.labels['input_ids'][idx].clone().detach()
            if self.tokenizer is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            item['labels'] = labels
            
        return item
    
    def __len__(self):
        return len(next(iter(self.encoder_input.values()))) 