from pathlib import Path
import os
import wget
import tarfile
import zipfile
from typing import Tuple
import pandas as pd

class DataManager:
    """데이터 다운로드, 로드, 전처리를 관리하는 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.data_path = Path(config.general.data_path)
        
    def prepare_data(self) -> None:
        """데이터 준비 (다운로드 및 압축해제)"""
        if not self.data_path.exists():
            print("Downloading data...")
            self.data_path.mkdir(parents=True, exist_ok=True)
            
            # config에서 URL 가져오기
            self._download_and_extract(self.config.url.data)
            self._download_and_extract(self.config.url.code)
            print("Data preparation completed")
        else:
            print(f"Data directory already exists at {self.data_path}")
            
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """학습/검증/테스트 데이터 로드"""
        self.prepare_data()  # 데이터 없으면 다운로드
        
        train_df = pd.read_csv(self.data_path / "train.csv")
        val_df = pd.read_csv(self.data_path / "dev.csv")
        test_df = pd.read_csv(self.data_path / "test.csv")
        
        return train_df, val_df, test_df
        
    def get_few_shot_samples(self, train_df: pd.DataFrame, n_samples: int = 1) -> pd.DataFrame:
        """few-shot 학습용 샘플 추출"""
        return train_df.sample(n=n_samples, random_state=self.config.general.seed)
            
    def _download_and_extract(self, url: str) -> None:
        """파일 다운로드 및 압축 해제"""
        filename = os.path.basename(url)
        download_path = self.data_path / filename
        
        # 다운로드
        print(f"Downloading {url}...")
        wget.download(url, str(download_path))
        print(f"\nDownloaded to {download_path}")
        
        # 압축 해제
        print("Extracting files...")
        if filename.endswith((".tar.gz", ".tgz")):
            with tarfile.open(download_path, "r:gz") as tar:
                tar.extractall(self.data_path)
        elif filename.endswith(".zip"):
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(self.data_path)
                
        # 압축파일 삭제
        download_path.unlink()
        print(f"Extracted to {self.data_path} and removed {filename}") 