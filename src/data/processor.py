from typing import Optional
import torch
from .dataset import DialogueDataset
from .data_manager import DataManager
import pandas as pd

class DataProcessor:
    """데이터 전처리 및 데이터셋 생성"""
    
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.data_manager = DataManager(config)
        
    def prepare_dataset(self, mode: str = "train") -> DialogueDataset:
        """데이터셋 준비"""
        # 데이터 로드
        data_path = self.config.general.data_path / f"{mode}.csv"
        df = pd.read_csv(data_path)
        
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