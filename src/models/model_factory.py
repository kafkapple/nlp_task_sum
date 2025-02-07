from pathlib import Path
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from omegaconf import DictConfig
import os
from dotenv import load_dotenv
from huggingface_hub import login

class ModelFactory:
    """Factory class for creating and managing various LLM models"""
    
    _model_families = {
        "solar": {
            "class": "SolarAPI",
            "module": "solar",
            "types": ["prompt"]
        },
        "bart": {
            "class": "BartSummarizer",
            "module": "bart",
            "types": ["finetune"]
        },
        "llama": {
            "class": "LlamaModel",
            "module": "llama",
            "types": ["finetune", "prompt"]
        }
    }
    
    def __init__(self):
        load_dotenv()  # .env 파일 로드
        if 'HUGGINGFACE_TOKEN' in os.environ:
            login(token=os.environ['HUGGINGFACE_TOKEN'])
    
    @classmethod
    def create_model(cls, config: DictConfig):
        """Create appropriate model instance based on config"""
        # huggingface 토큰 설정
        os.environ['HUGGINGFACE_TOKEN'] = config.huggingface.token
        login(token=config.huggingface.token)
        
        model_family = config.model.family
        model_type = config.model.type

        if model_family not in cls._model_families:
            raise ValueError(f"Unsupported model family: {model_family}")
            
        model_info = cls._model_families[model_family]
        if model_type not in model_info["types"]:
            raise ValueError(f"Model {model_family} does not support type: {model_type}")

        # 동적으로 모델 클래스 import
        module = __import__(f"src.models.{model_info['module']}", 
                          fromlist=[model_info['class']])
        model_class = getattr(module, model_info['class'])
        
        return model_class(config)

    @classmethod
    def create_model_and_tokenizer(cls, config: DictConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Create or load raw model and tokenizer"""
        # 상위 디렉토리의 models 폴더 사용
        model_dir = Path(config.huggingface.cache_dir) / config.model.name.split('/')[-1]
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 양자화 설정
        bnb_config = cls._create_quantization_config(config)
        
        kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "quantization_config": bnb_config,
            "cache_dir": str(model_dir),  # config의 cache_dir 사용
            "use_auth_token": config.huggingface.token
        }
        
        if (model_dir / "model").exists() and not config.model.force_download:
            print(f"Loading model from {model_dir}")
            model = cls._load_model(str(model_dir / "model"), bnb_config, cache_dir=str(model_dir))
            tokenizer = cls._load_tokenizer(str(model_dir / "tokenizer"))
        else:
            print(f"Downloading model {config.model.name}...")
            model = cls._load_model(config.model.name, bnb_config)
            tokenizer = cls._load_tokenizer(config.model.name)
            
            # 모델과 토크나이저 저장
            model.save_pretrained(str(model_dir / "model"))
            tokenizer.save_pretrained(str(model_dir / "tokenizer"))
            
        return model, tokenizer

    @staticmethod
    def _create_quantization_config(config: DictConfig) -> Optional[BitsAndBytesConfig]:
        """Create quantization configuration"""
        if not config.quantization.enabled:
            return None
            
        return BitsAndBytesConfig(
            load_in_8bit=config.quantization.precision == "int8",
            load_in_4bit=config.quantization.precision == "int4",
            bnb_4bit_compute_dtype=getattr(torch, config.quantization.compute_dtype),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    @staticmethod
    def _load_model(model_path: str, quantization_config: Optional[BitsAndBytesConfig] = None, cache_dir: str = "") -> AutoModelForCausalLM:
        """Load model with fallback options"""
        kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "quantization_config": quantization_config,
            "cache_dir": cache_dir,  # 캐시 디렉토리 설정
            "use_auth_token": os.getenv('HUGGINGFACE_TOKEN')  # 인증 토큰 사용
        }
        
        try:
            print("float16으로 모델 로딩 시도...")
            return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        except Exception as e:
            print(f"float16 로딩 실패: {str(e)}")
            try:
                print("float32로 다시 시도...")
                kwargs["torch_dtype"] = torch.float32
                return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            except Exception as e:
                print(f"float32로도 실패: {str(e)}")
                try:
                    print("CPU로 마지막 시도...")
                    kwargs["device_map"] = "cpu"
                    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
                except Exception as e:
                    print(f"모든 시도 실패: {str(e)}")
                    raise

    @staticmethod
    def _load_tokenizer(model_path: str) -> AutoTokenizer:
        """Load tokenizer with proper settings"""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer 