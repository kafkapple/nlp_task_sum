import hydra
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
from src.data.dataset import DataProcessor
from src.models.bart import DialogueSummarizerConfig, TokenizerConfig, BartSummarizer
from omegaconf import DictConfig
import os
import pandas as pd

class DialogueInference:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.model_config = self._create_model_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 추론 최적화
        if self.device == "cuda":
            self.model = self.model.half()  # FP16으로 변환
        self.model.to(self.device)
        self.model.eval()
        torch.set_grad_enabled(False)
        
    def _load_model_and_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
            special_tokens = [str(token) for token in self.cfg.model.tokenizer.special_tokens.additional_special_tokens]
            special_tokens_dict = {'additional_special_tokens': special_tokens}
            tokenizer.add_special_tokens(special_tokens_dict)
            
            # 체크포인트 경로 처리
            if os.path.exists(self.cfg.inference.ckt_path):
                # 직접 경로가 있는 경우
                model_path = self.cfg.inference.ckt_path
            else:
                # outputs 디렉토리에서 가장 최근 체크포인트 찾기
                outputs_dir = os.path.join(os.getcwd(), 'outputs')
                if not os.path.exists(outputs_dir):
                    raise ValueError(f"No outputs directory found at {outputs_dir}")
                    
                # 가장 최근 타임스탬프 폴더 찾기
                timestamps = sorted([d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))])
                if not timestamps:
                    raise ValueError("No timestamp directories found in outputs/")
                    
                latest_dir = os.path.join(outputs_dir, timestamps[-1])
                
                # checkpoint 폴더 찾기
                checkpoints = sorted([d for d in os.listdir(latest_dir) if d.startswith('checkpoint-')])
                if not checkpoints:
                    raise ValueError(f"No checkpoints found in {latest_dir}")
                    
                model_path = os.path.join(latest_dir, checkpoints[-1])
                print(f"Using latest checkpoint: {model_path}")
            
            # 모델 로드
            model = BartForConditionalGeneration.from_pretrained(model_path)
            model.resize_token_embeddings(len(tokenizer))
            return model, tokenizer
        
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def _create_model_config(self):
        return DialogueSummarizerConfig(
            model_name=self.cfg.model.name,
            tokenizer=TokenizerConfig(
                bos_token=self.cfg.model.tokenizer.bos_token,
                eos_token=self.cfg.model.tokenizer.eos_token,
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
                    max_length=self.cfg.inference.generate_max_length,
                    num_beams=self.cfg.inference.num_beams,
                    use_cache=True  # 캐시 사용으로 속도 향상
                )
                
                decoded_summaries = self.tokenizer.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )
                summaries.extend(decoded_summaries)
                
        return summaries
    
    def inference(self, test_file_path):
        # 데이터 준비
        processor = DataProcessor(self.tokenizer, self.model_config)
        test_dataset = processor.prepare_data(test_file_path, is_train=False)
        
        # 요약문 생성
        summaries = self.generate_summaries(test_dataset)
        
        # 결과 저장
        test_df = pd.read_csv(test_file_path)
        results = pd.DataFrame({
            'fname': test_df['fname'],
            'summary': summaries
        })
        
        os.makedirs(self.cfg.general.output_path, exist_ok=True)
        output_path = f"{self.cfg.general.output_path}/predictions.csv"
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        return results

@hydra.main(version_base="1.2", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    inferencer = DialogueInference(cfg)
    test_file_path = f"{cfg.general.data_path}/test.csv"
    results = inferencer.inference(test_file_path)

if __name__ == "__main__":
    main() 