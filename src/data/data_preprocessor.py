import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

class DataPreprocessor:
    def __init__(self, config):
        """
        Args:
            config: 전체 설정 객체
        """
        self.config = config
        self.data_path = Path(config.general.data_path)
        self.preprocessing_config = config.train.preprocessing
        
    def _calculate_text_lengths(self, df):
        """텍스트 길이 계산"""
        df['dialogue_len'] = df['dialogue'].str.len()
        df['summary_len'] = df['summary'].str.len()
        return df
    
    def _balance_length_distribution(self, df):
        """대화/요약 길이 분포 균형 맞추기"""
        length_strategy = self.preprocessing_config.get('length_strategy', 'dialogue')
        length_bins = self.preprocessing_config.get('length_bins', {
            'dialogue': 5,
            'summary': 5,
            'ratio': 5
        })
        
        def create_labels(n):
            """구간 수에 따른 레이블 생성"""
            return [str(i+1) for i in range(n)]
        
        if length_strategy == 'dialogue':
            # 대화 길이만 사용
            n_bins = length_bins.get('dialogue', 5)
            df['length_bin'] = pd.qcut(
                df['dialogue_len'], 
                q=n_bins, 
                labels=create_labels(n_bins)
            )
            
        elif length_strategy == 'summary':
            # 요약 길이만 사용
            n_bins = length_bins.get('summary', 5)
            df['length_bin'] = pd.qcut(
                df['summary_len'], 
                q=n_bins, 
                labels=create_labels(n_bins)
            )
            
        elif length_strategy == 'both':
            # 대화와 요약 길이 모두 고려
            d_bins = length_bins.get('dialogue', 3)
            s_bins = length_bins.get('summary', 3)
            
            df['dialogue_bin'] = pd.qcut(
                df['dialogue_len'], 
                q=d_bins, 
                labels=create_labels(d_bins)
            )
            df['summary_bin'] = pd.qcut(
                df['summary_len'], 
                q=s_bins, 
                labels=create_labels(s_bins)
            )
            df['length_bin'] = df['dialogue_bin'] + '_' + df['summary_bin']
            df.drop(columns=['dialogue_bin', 'summary_bin'], inplace=True)
            
        elif length_strategy == 'ratio':
            # 요약/대화 길이 비율 사용
            n_bins = length_bins.get('ratio', 5)
            df['length_ratio'] = df['summary_len'] / df['dialogue_len']
            df['length_bin'] = pd.qcut(
                df['length_ratio'], 
                q=n_bins, 
                labels=create_labels(n_bins)
            )
            df.drop(columns=['length_ratio'], inplace=True)
        
        # 분포 출력
        print(f"\n=== Length Distribution ({length_strategy}) ===")
        print(df['length_bin'].value_counts().sort_index())
        
        return df
    
    def _get_stratify_columns(self, df):
        """층화 추출을 위한 컬럼 준비"""
        stratify_cols = []
        
        # 설정된 stratify_column이 있으면 추가
        if self.preprocessing_config.stratify_column:
            if self.preprocessing_config.stratify_column in df.columns:
                stratify_cols.append(self.preprocessing_config.stratify_column)
        
        # 길이 기반 층화도 사용
        if self.preprocessing_config.get('stratify_by_length', False):
            stratify_cols.append('length_bin')
            
        # 여러 컬럼으로 층화할 경우 조합
        if len(stratify_cols) > 1:
            return df[stratify_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        elif len(stratify_cols) == 1:
            return df[stratify_cols[0]]
        return None

    def prepare_data(self):
        """데이터 전처리 및 분할 수행"""
        if not self.preprocessing_config.enabled:
            print("데이터 전처리가 비활성화되어 있습니다. 기존 데이터를 사용합니다.")
            return self.data_path
            
        # 데이터 로드 및 병합
        train_df = pd.read_csv(self.data_path / "train.csv")
        
        # processed 폴더에서 먼저 찾고, 없으면 원본 경로에서 찾기
        processed_val = self.data_path / "processed" / "validation.csv"
        original_dev = self.data_path / "dev.csv"
        
        if processed_val.exists():
            dev_df = pd.read_csv(processed_val)
        else:
            dev_df = pd.read_csv(original_dev)

        if self.preprocessing_config.merge_train_dev:
            print("train과 dev 데이터를 병합합니다.")
            combined_df = pd.concat([train_df, dev_df], ignore_index=True)
            
            # 중복 제거
            combined_df = combined_df.drop_duplicates(subset=['fname', 'dialogue'])
            combined_df = combined_df.dropna()
            
            # 텍스트 길이 계산 및 분포 균형
            combined_df = self._calculate_text_lengths(combined_df)
            combined_df = self._balance_length_distribution(combined_df)
            
            # 층화 기준 설정
            stratify = self._get_stratify_columns(combined_df)
            
            try:
                # 데이터 분할
                train_data, val_data = train_test_split(
                    combined_df,
                    test_size=self.preprocessing_config.split_ratio,
                    stratify=stratify,
                    random_state=self.preprocessing_config.random_state
                )
                
                # 분할 결과 통계
                print("\n=== 데이터 분할 통계 ===")
                print(f"Train 데이터: {len(train_data)}개")
                print(f"Validation 데이터: {len(val_data)}개")
                
                # stratify_column이 있을 때만 해당 통계 출력
                if self.preprocessing_config.stratify_column:
                    print("\n클래스별 분포:")
                    print("Train:", train_data[self.preprocessing_config.stratify_column].value_counts())
                    print("Val:", val_data[self.preprocessing_config.stratify_column].value_counts())
                
                # 길이 기반 층화를 사용할 때는 length_bin 분포 출력
                if self.preprocessing_config.stratify_by_length:
                    print("\n길이 구간별 분포:")
                    print("Train:", train_data['length_bin'].value_counts().sort_index())
                    print("Val:", val_data['length_bin'].value_counts().sort_index())
                
                # 길이 통계
                print("\n텍스트 길이 통계:")
                print("Train - 대화 길이:", train_data['dialogue_len'].describe())
                print("Train - 요약 길이:", train_data['summary_len'].describe())
                print("Val - 대화 길이:", val_data['dialogue_len'].describe())
                print("Val - 요약 길이:", val_data['summary_len'].describe())
                
                if self.preprocessing_config.length_strategy == 'ratio':
                    ratio_train = train_data['summary_len'] / train_data['dialogue_len']
                    ratio_val = val_data['summary_len'] / val_data['dialogue_len']
                    print("\n요약/대화 길이 비율 통계:")
                    print("Train - 비율:", ratio_train.describe())
                    print("Val - 비율:", ratio_val.describe())
                
            except ValueError as e:
                print(f"층화 추출 실패: {e}")
                print("일반 랜덤 분할로 진행합니다.")
                train_data, val_data = train_test_split(
                    combined_df,
                    test_size=self.preprocessing_config.split_ratio,
                    stratify=None,
                    random_state=self.preprocessing_config.random_state
                )
            
            # 전처리 컬럼 제거
            for df in [train_data, val_data]:
                df.drop(columns=['dialogue_len', 'summary_len', 'length_bin'], 
                       errors='ignore', inplace=True)
            
            # 저장
            processed_dir = self.data_path / "processed"
            processed_dir.mkdir(exist_ok=True)
            
            # train, dev 데이터 저장
            train_data.to_csv(processed_dir / "train.csv", index=False)
            val_data.to_csv(processed_dir / "dev.csv", index=False)
            
            # test 데이터 복사
            test_src = self.data_path / "test.csv"
            test_dst = processed_dir / "test.csv"
            if test_src.exists():
                shutil.copy2(test_src, test_dst)
                print(f"test.csv를 {processed_dir}에 복사했습니다.")
            
            return processed_dir
        
        return self.data_path

if __name__ == "__main__":
    processor = DataPreprocessor()
    processor.prepare_data()