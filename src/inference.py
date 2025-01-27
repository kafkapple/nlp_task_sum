import hydra
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    T5ForConditionalGeneration
)
from src.data.dataset import DataProcessor
from src.models.bart import DialogueSummarizerConfig, TokenizerConfig, BartSummarizer
from omegaconf import DictConfig
import os
import pandas as pd

class DialogueInference:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.model_config = self._create_model_config()
        
        # 추론 최적화
        if self.device == "cuda":
            self.model = self.model.half()  # FP16으로 변환
        self.model.to(self.device)
        self.model.eval()
        torch.set_grad_enabled(False)
        
    def _load_model_and_tokenizer(self):
        """모델과 토크나이저 로드"""
        try:
            # 체크포인트 경로 처리
            if os.path.exists(self.cfg.inference.ckt_path):
                # 직접 경로가 있는 경우
                model_path = self.cfg.inference.ckt_path
            else:
                # outputs 디렉토리에서 가장 최근 체크포인트 찾기
                outputs_dir = os.path.join(os.getcwd(), 'outputs')
                if not os.path.exists(outputs_dir):
                    # 체크포인트가 없으면 원본 모델 사용
                    model_path = self.cfg.model.name
                else:
                    # 최신 체크포인트 찾기
                    dirs = sorted([d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))])
                    if not dirs:
                        model_path = self.cfg.model.name
                    else:
                        latest_dir = os.path.join(outputs_dir, dirs[-1])
                        checkpoints = [d for d in os.listdir(latest_dir) if d.startswith('checkpoint-')]
                        if not checkpoints:
                            model_path = self.cfg.model.name
                        else:
                            model_path = os.path.join(latest_dir, sorted(checkpoints)[-1])

            print(f"Loading model from: {model_path}")
            
            # 모델 타입에 따라 적절한 클래스 선택
            if "t5" in self.cfg.model.name.lower():
                model = T5ForConditionalGeneration.from_pretrained(model_path)
            else:
                model = BartForConditionalGeneration.from_pretrained(model_path)
            
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
            
            return model, tokenizer
        
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # 원본 모델로 폴백
            return self._load_original_model()

    def _load_original_model(self):
        """원본 모델 로드"""
        print(f"Loading original model: {self.cfg.model.name}")
        if "t5" in self.cfg.model.name.lower():
            model = T5ForConditionalGeneration.from_pretrained(self.cfg.model.name)
        else:
            model = BartForConditionalGeneration.from_pretrained(self.cfg.model.name)
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        return model, tokenizer
    
    def _create_model_config(self):
        return DialogueSummarizerConfig(
            model_name=self.cfg.model.name,
            tokenizer=TokenizerConfig(
                bos_token=getattr(self.cfg.model.tokenizer, 'bos_token', '<s>'),
                eos_token=getattr(self.cfg.model.tokenizer, 'eos_token', '</s>'),
                decoder_max_len=self.cfg.model.tokenizer.decoder_max_len,
                encoder_max_len=self.cfg.model.tokenizer.encoder_max_len,
                special_tokens=self.cfg.model.tokenizer.special_tokens
            )
        )
    
    def generate_summaries(self, test_dataset):
        dataloader = DataLoader(
            test_dataset, 
            batch_size=self.cfg.inference.batch_size,
            num_workers=4,  # 데이터 로딩 병렬화
            pin_memory=True  # CUDA 전송 최적화
        )
        
        summaries = []
        with torch.cuda.amp.autocast():  # Mixed precision inference
            for batch in tqdm(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    no_repeat_ngram_size=self.cfg.inference.no_repeat_ngram_size,
                    early_stopping=self.cfg.inference.early_stopping,
                    max_new_tokens=self.cfg.inference.max_new_tokens,
                    min_new_tokens=self.cfg.inference.min_new_tokens,
                    num_beams=self.cfg.inference.num_beams,
                    use_cache=True  # 캐시 사용으로 속도 향상
                )
                
                decoded_summaries = self.tokenizer.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )
                summaries.extend(decoded_summaries)
                
        return summaries
    
    def inference(self, data_path):
        """
        Args:
            data_path: 데이터 디렉토리 경로
        """
        # 데이터 경로 확인 및 처리
        data_path = os.path.abspath(data_path)
        print(f"Using data path: {data_path}")
        
        if not os.path.exists(data_path):
            raise ValueError(f"Data directory not found at {data_path}")
        
        # 데이터 준비
        processor = DataProcessor(
            tokenizer=self.tokenizer, 
            config=self.cfg.model,
            data_path=data_path
        )
        
        # test.csv 파일 경로 생성
        test_file_path = os.path.join(data_path, "test.csv")
        if not os.path.exists(test_file_path):
            raise ValueError(f"Test file not found at {test_file_path}")
        
        # test 데이터셋 준비 (summary 컬럼이 없는 경우 처리)
        test_df = pd.read_csv(test_file_path)
        if 'summary' not in test_df.columns:
            test_df['summary'] = ''  # 빈 문자열로 채움
            # 수정된 DataFrame을 임시 파일로 저장
            processed_test_path = os.path.join(data_path, "processed_test.csv")
            test_df.to_csv(processed_test_path, index=False)
            data_path = os.path.dirname(processed_test_path)  # 새로운 데이터 경로 설정
        
        test_dataset = processor.prepare_dataset("processed_test" if 'summary' not in test_df.columns else "test")
        
        # 요약문 생성
        summaries = self.generate_summaries(test_dataset)
        
        # 결과 저장
        results = pd.DataFrame({
            'fname': test_df['fname'],
            'summary': summaries
        })
        
        os.makedirs(self.cfg.general.output_path, exist_ok=True)
        output_path = f"{self.cfg.general.output_path}/predictions.csv"
        results.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {os.path.abspath(output_path)}")
        print(f"File exists: {os.path.exists(output_path)}")
        print(f"File size: {os.path.getsize(output_path) if os.path.exists(output_path) else 'N/A'} bytes")
        
        return results

@hydra.main(version_base="1.2", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    inferencer = DialogueInference(cfg)
    test_file_path = f"{cfg.general.data_path}/test.csv"
    results = inferencer.inference(test_file_path)

if __name__ == "__main__":
    main() 