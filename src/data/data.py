import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import logging

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
            
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train, validation and test datasets"""
        files = ['train.csv', 'dev.csv', 'test.csv']
        for file in files:
            if not (self.data_path / file).exists():
                raise FileNotFoundError(f"Required file not found: {self.data_path / file}")
                
        train_df = pd.read_csv(self.data_path / 'train.csv')
        val_df = pd.read_csv(self.data_path / 'dev.csv')
        test_df = pd.read_csv(self.data_path / 'test.csv')
        
        logging.info(f"Loaded data from {self.data_path}")
        logging.info(f"Train samples: {len(train_df)}")
        logging.info(f"Validation samples: {len(val_df)}")
        logging.info(f"Test samples: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_few_shot_samples(self, train_df: pd.DataFrame, n_samples: int = 1, random_seed: int = 2001) -> pd.DataFrame:
        """Get few shot samples from training data"""
        return train_df.sample(n_samples, random_state=random_seed) 