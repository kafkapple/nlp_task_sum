import pandas as pd
import os
import time
from tqdm import tqdm
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.
from openai import OpenAI # openai==1.2.0
from dotenv import load_dotenv
import wandb
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
import shutil

from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
import openai
from src.utils.wandb_logger import WandBLogger
from src.utils.metrics import MetricsCalculator

def log_rouge(scores):
    """ROUGE 점수를 wandb에 기록"""
    dict_rouge = {
        "rouge1_f1": "eval/rouge1_f1",
        "rouge2_f1": "eval/rouge2_f1",
        "rougeL_f1": "eval/rougeL_f1",
    }
    for key, value in scores.items():
        if key in dict_rouge and wandb.run is not None:
            wandb.log({dict_rouge[key]: value})
        print(f"{key}: {value}")

def convert_to_basic_types(obj):
    """OmegaConf 설정을 기본 Python 타입으로 변환"""
    if isinstance(obj, dict):
        return {k: convert_to_basic_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_basic_types(v) for v in obj]
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif obj is None:
        return None
    else:
        return str(obj)

def init_logger(cfg):
    """Initialize wandb logger"""
    try:
        # 이전 wandb 프로세스 정리
        if wandb.run is not None:
            wandb.finish()
            
        timestamp = cfg.general.timestamp
        run_name = f"{timestamp}"
        
        # output_path 생성
        output_dir = Path(cfg.general.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # config 폴더 복사
        config_dir = Path("config")
        output_config_dir = output_dir / "config"
        if config_dir.exists():
            if output_config_dir.exists():
                shutil.rmtree(output_config_dir)
            shutil.copytree(config_dir, output_config_dir)
            print(f"Copied config files to {output_config_dir}")
        
        # wandb 설정
        os.environ["WANDB_DIR"] = str(output_dir)
        os.environ["WANDB_START_METHOD"] = "thread"
        os.environ["WANDB_WATCH"] = "false"  # 모델 가중치 로깅 비활성화
        
        # 설정을 기본 Python 타입으로 변환
        config_dict = convert_to_basic_types(OmegaConf.to_container(cfg, resolve=True))
        
        # wandb 초기화
        if hasattr(cfg.general, 'wandb'):
            wandb.init(
                project=cfg.general.wandb.project,
                name=run_name,
                config=config_dict,
                group=cfg.model.family,
                dir=str(output_dir),
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True,
                    _disable_meta=True
                ),
                mode="online",
                reinit=True
            )
        else:
            print("wandb 설정을 찾을 수 없습니다. wandb 로깅 없이 실행합니다.")
            
        return output_dir
        
    except Exception as e:
        print(f"wandb 초기화 실패: {str(e)}")
        print("wandb 로깅 없이 계속 실행합니다...")
        return Path(cfg.general.output_path)

load_dotenv()


def compute_metrics(pred, gold):
    # 모델 성능에 대한 평가 지표를 정의합니다. 본 대회에서는 ROUGE 점수를 통해 모델의 성능을 평가합니다.    
    rouge = Rouge()
    results = rouge.get_scores(pred, gold, avg=True)
    result = {key: value["f"] for key, value in results.items()}
    return result
# Few-shot sample을 다른 방식으로 사용하여 prompt를 생성합니다.
def build_prompt(config, dialogue, sample_dialogue1, sample_summary1):
    system_prompt = "You are a expert in the field of dialogue summarization, summarize the given dialogue in a concise manner. Follow the user's instruction carefully and provide a summary that is relevant to the dialogue."

    few_shot_user_prompt_1 = (
        "Following the instructions below, summarize the given document.\n"
        "Instructions:\n"
        "1. Read the provided sample dialogue and corresponding summary.\n"
        "2. Read the dialogue carefully.\n"
        "3. Following the sample's style of summary, provide a concise summary of the given dialogue. Be sure that the summary is simple but captures the essence of the dialogue.\n\n"
        "Dialogue:\n"
        f"{sample_dialogue1}\n\n"
        "Summary:\n"
    )
    few_shot_assistant_prompt_1 = sample_summary1
    
    user_prompt = (
        "Dialogue:\n"
        f"{dialogue}\n\n"
        "Summary:\n"
    )
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": few_shot_user_prompt_1},
        {"role": "assistant", "content": few_shot_assistant_prompt_1},
        {"role": "user", "content": user_prompt},
    ]
    print("\n=======\n",prompt)
    return prompt
#Dialogue를 입력으로 받아, Solar Chat API에 보낼 Prompt를 생성하는 함수를 정의합니다.
# def build_prompt(config, dialogue, sample_dialogues, sample_summaries):
#     if config["custom_config"]["few_shot"]:
#         system_prompt = (
#             "You are a expert in the field of dialogue summarization, summarize the given dialogue in a concise manner. Follow the user's instruction carefully and provide a summary that is relevant to the dialogue.\n"

#             Convert colloquial language into formal written language"
#         )

#         # 여러 few-shot 예제를 하나의 문자열로 결합
#         few_shot_examples = ""
#         if isinstance(sample_dialogues, list) and isinstance(sample_summaries, list):
#             for d, s in zip(sample_dialogues, sample_summaries):
#                 few_shot_examples += f"대화:\n{d}\n요약:\n{s}\n\n"
#         else:
#             few_shot_examples = f"대화:\n{sample_dialogues}\n요약:\n{sample_summaries}\n\n"

#         user_prompt = (
#         "Following the instructions below, summarize the given document.\n"
#         "Instructions:\n"
#         "1. Read the provided sample dialogue and corresponding summary.\n"
#         "2. Read the dialogue carefully.\n"
#         "3. Following the sample's style of summary, provide a concise summary of the given dialogue. Be sure that the summary is simple but captures the essence of the dialogue.\n\n"
#             f"{few_shot_examples}\n"
#             "이제 아래 대화를 위 예시들과 비슷한 스타일로 요약해주세요.\n"
#             f"대화:\n{dialogue}\n\n"
#             "요약:"
#         )
        
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ]
#     else:
#         system_prompt = (
#             "You are a Korean dialogue summarization expert.\n"
#             "Create a concise and coherent summary that captures the key points of the dialogue.\n"
#             "The summary should be in Korean and maintain the original context."
#         )
#         user_prompt = f"대화:\n{dialogue}\n\n요약:"
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ]
    
#     return messages
def summarization(config, dialogue, client, sample_dialogues, sample_summaries):
    messages = build_prompt(config, dialogue, sample_dialogues, sample_summaries)
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            summary = client.chat.completions.create(
                model="solar-1-mini-chat",
                messages=messages,
                temperature=config["custom_config"]["temperature"],
                top_p=config["custom_config"]["top_p"],
            )
            return summary.choices[0].message.content
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                raise
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(retry_delay)
            retry_delay *= 2  # exponential backoff

def get_similar_length_sample(train_df, target_length, n_candidates=5):
    """입력 텍스트 길이와 가장 비슷한 대화를 train_df에서 찾아 반환"""
    train_df['length_diff'] = abs(train_df['dialogue_len'] - target_length)
    similar_samples = train_df.nsmallest(n_candidates, 'length_diff')
    # 후보들 중 랜덤 선택
    selected_sample = similar_samples.sample(n=1).iloc[0]
    return selected_sample['dialogue'], selected_sample['summary']

def validate(config, val_df, client, sample_dialogues, sample_summaries, num_samples=-1):
    """Validation data의 대화를 요약하고, 점수를 측정합니다."""
    logger = WandBLogger(config)
    metrics_calc = MetricsCalculator()
    
    predictions = []
    references = []
    scores = []
    raw_scores = []
    
    # Rate limiting을 위한 변수들
    requests_per_minute = 50
    start_time = time.time()
    request_count = 0
    
    val_samples = val_df[:num_samples] if num_samples > 0 else val_df
    
    for idx, row in tqdm(val_samples.iterrows(), total=len(val_samples)):
        request_count += 1
        if request_count >= requests_per_minute:
            elapsed = time.time() - start_time
            if elapsed < 60:
                sleep_time = 60 - elapsed + 5
                print(f"\nWaiting {sleep_time:.1f} seconds to respect rate limit...")
                time.sleep(sleep_time)
            start_time = time.time()
            request_count = 0
        
        try:
            dialogue = row['dialogue']
            current_sample_dialogues = sample_dialogues
            current_sample_summaries = sample_summaries
            
            # similar_length 모드일 경우 대화 길이 기반으로 few-shot sample 선택
            if config.custom_config.few_shot_selection == "similar_length":
                dialogue_len = len(dialogue)
                current_sample_dialogues, current_sample_summaries = get_similar_length_sample(
                    val_df, 
                    dialogue_len,
                    config.custom_config.similar_length_selection.n_candidates
                )
            
            # 예측 수행
            pred = summarization(config, dialogue, client, current_sample_dialogues, current_sample_summaries)
            predictions.append(pred)
            references.append(row['summary'])
            
            # ROUGE 점수 계산
            results = compute_metrics(pred, row['summary'])
            avg_score = sum(results.values()) / len(results)
            scores.append(avg_score)
            raw_scores.append(results)
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {str(e)}")
            continue
    
    # 메트릭 계산
    predictions_df, metrics = metrics_calc.compute_rouge_scores(predictions, references)
    
    # 결과 저장 및 로깅
    save_path = os.path.join(config.general.output_path, "validation_samples.csv")
    logger.log_predictions(predictions_df, save_path=save_path)
    logger.log_metrics(metrics)
    
    val_avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\nValidation Average Score: {val_avg_score}")
    
    return metrics

def inference(config, client,test_df,sample_dialogues,sample_summaries):
    print("======= Inference Start... ========")
    
    summary = []
    start_time = time.time()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        dialogue = row['dialogue']
        summary.append(summarization(config, dialogue,client,sample_dialogues,sample_summaries))
        
        # Rate limit 방지를 위해 1분 동안 최대 100개의 요청을 보내도록 합니다.
        if (idx + 1) % 100 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if elapsed_time < 60:
                wait_time = 60 - elapsed_time + 5
                print(f"Elapsed time: {elapsed_time:.2f} sec")
                print(f"Waiting for {wait_time} sec")
                time.sleep(wait_time)
            
            start_time = time.time()
    
    output = pd.DataFrame(
        {
            "fname": test_df['fname'],
            "summary" : summary,
        }
    )
    RESULT_PATH = config.general.output_path
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    output.to_csv(os.path.join(RESULT_PATH, "output_solar.csv"), index=False)

    return output

def calc_train_score(config, train_df,client, sample_dialogue1, sample_summary1,num_samples=-1):
    train_samples = train_df[:num_samples] if num_samples > 0 else train_df
    
    scores = []
    raw_scores = []
    for idx, row in tqdm(train_samples.iterrows(), total=len(train_samples)):
        dialogue = row['dialogue']
        summary = summarization(config, dialogue, client, sample_dialogue1, sample_summary1)
        results = compute_metrics(summary, row['summary'])
        avg_score = sum(results.values()) / len(results)
        
        scores.append(avg_score)
        raw_scores.append(results)
        for key, value in results.items():
            train_samples[f'{key}_score'] = value
        
    train_avg_score = sum(scores) / len(scores)

    print(f"Train Average Score: {train_avg_score}")
    return raw_scores

# Train data 중 처음 3개의 대화를 요약합니다.
def test_on_train_data(config, train_df,client, sample_dialogue1, sample_summary1,num_samples=3):

    train_df_sample = train_df[:num_samples]
    train_df_rest = train_df[num_samples:]
    print(f"train_df describe: {train_df_rest.describe()}")
    print(f"train_df_sample describe: {train_df_sample.describe()}")
    for idx, row in train_df_sample.iterrows():
        dialogue = row['dialogue']
        summary = summarization(config, dialogue, client, sample_dialogue1, sample_summary1)
        print(f"Dialogue:\n{dialogue}\n")
        print(f"Pred Summary: {summary}\n")
        print(f"Gold Summary: {row['summary']}\n")
        print("=="*50)

def eda(df):
    try:
        df['dialogue_len'] = df.dialogue.str.len()
    except:
        print("data has no dialogue column")
    try:
        df['summary_len'] = df.summary.str.len()
    except:
        print(" data has no summary column")
    try:
        # agg_df = df.agg(['mean','std','min','max'])
        # print(agg_df)
        print(df.describe())
    except:
        print("data has no fname column")
    

@hydra.main(version_base="1.3", config_path="config", config_name="config_solar")
def main(cfg: DictConfig):
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
    DATA_PATH = cfg.general.data_path

    random_seed = cfg.general.seed
    n_val_samples = cfg.custom_config.n_val_samples
    cfg.model.name = "solar-0"

    wandb_logger = init_logger(cfg)
    # train data의 구조와 내용을 확인합니다.
    train_df = pd.read_csv(os.path.join(DATA_PATH,'train.csv'))
    # validation data의 구조와 내용을 확인합니다.
    val_df = pd.read_csv(os.path.join(DATA_PATH,'dev.csv'))
    print("===eda train_df===")
    eda(train_df)
    print("===eda val_df===")
    eda(val_df)
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    eda(test_df)
    # Few-shot prompt를 생성하기 위해, train data의 일부를 사용합니다.

    if cfg.custom_config.few_shot_selection == "similar_length":
        # 초기 few-shot sample은 첫 번째 검증 데이터의 길이와 비슷한 것으로 선택
        first_val_len = len(val_df.iloc[0]['dialogue'])
        sample_dialogue, sample_summary = get_similar_length_sample(
            train_df,
            first_val_len,
            cfg.custom_config.similar_length_selection.n_candidates
        )
    elif cfg.custom_config.few_shot_selection == "max_len_summary":
        few_shot_samples = train_df.nlargest(cfg.custom_config.n_few_shot_samples, "summary_len")
        sample_dialogue = few_shot_samples.iloc[0]['dialogue']
        sample_summary = few_shot_samples.iloc[0]['summary']
    elif cfg.custom_config.few_shot_selection == "min_max_len_summary":
        few_shot_sample_min = train_df.nsmallest(cfg.custom_config.n_few_shot_samples, "summary_len")
        few_shot_sample_max = train_df.nlargest(cfg.custom_config.n_few_shot_samples, "summary_len")
        few_shot_samples = pd.concat([few_shot_sample_min, few_shot_sample_max])
        sample_dialogue = few_shot_samples.iloc[0]['dialogue']
        sample_summary = few_shot_samples.iloc[0]['summary']
    elif cfg.custom_config.few_shot_selection == "baseline":
        few_shot_samples = train_df.sample(cfg.custom_config.n_few_shot_samples, random_state=random_seed)
        sample_dialogue = few_shot_samples.iloc[0]['dialogue']
        sample_summary = few_shot_samples.iloc[0]['summary']
    elif cfg.custom_config.few_shot_selection == "idx":
        few_shot_samples = train_df.iloc[cfg.custom_config.n_few_shot_samples-1]
        print(few_shot_samples)
        sample_dialogue = few_shot_samples['dialogue']
        sample_summary = few_shot_samples['summary']
    else:
        raise ValueError(f"Invalid few_shot_selection value: {cfg.custom_config.few_shot_selection}")

    print(f"Sample Dialogue1:\n{sample_dialogue}\n")
    print(f"Sample Summary1: {sample_summary}\n")

    client = OpenAI(
        api_key=UPSTAGE_API_KEY,
        base_url="https://api.upstage.ai/v1/solar"
    )

    scores =validate(cfg,val_df,client,sample_dialogue,sample_summary,n_val_samples)
    print(scores)
    log_rouge(scores)

    if cfg.inference.enabled:
        print("======= Inference Start... ========")
        _ = inference(cfg,client,test_df,sample_dialogue,sample_summary)
    
    wandb.finish()

if __name__ == "__main__":
    main()