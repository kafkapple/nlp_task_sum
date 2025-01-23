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

def log_rouge(scores, wandb_logger):
    dict_rouge ={
        "rouge-1": "eval/rouge1_f1",
        "rouge-2": "eval/rouge2_f1",
        "rouge-l": "eval/rougeL_f1",
    }
    for key, value in scores.items():
        wandb_logger.experiment.log({dict_rouge[key]: value})
        print(f"{dict_rouge[key]}: {value}")

def init_logger(cfg):
    """Initialize wandb logger"""
    try:
        # 이전 wandb 프로세스 정리
        if wandb.run is not None:
            wandb.finish()
            
        timestamp = cfg.general.timestamp
        run_name = f"{cfg.model.name}_{cfg.model.mode}_{timestamp}"
        
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
            print("Warning: wandb configuration not found in config. Running without wandb logging.")
            
        return output_dir
        
    except Exception as e:
        print(f"Warning: wandb initialization failed: {str(e)}")
        print("Continuing without wandb logging...")
        return Path(cfg.general.output_path)

load_dotenv()


def compute_metrics(pred, gold):
    # 모델 성능에 대한 평가 지표를 정의합니다. 본 대회에서는 ROUGE 점수를 통해 모델의 성능을 평가합니다.    
    rouge = Rouge()
    results = rouge.get_scores(pred, gold, avg=True)
    result = {key: value["f"] for key, value in results.items()}
    return result

#Dialogue를 입력으로 받아, Solar Chat API에 보낼 Prompt를 생성하는 함수를 정의합니다.
def build_prompt(config, dialogue, sample_dialogues, sample_summaries):
    if config["custom_config"]["few_shot"]:
        system_prompt = (
            "You are a Korean dialogue summarization expert. Follow these guidelines:\n"
            "1. Create a concise and coherent summary in Korean\n"
            "2. Capture the key points and main context of the dialogue\n"
            "3. Maintain a natural flow in the summary\n"
            "4. Keep the tone consistent with the original dialogue\n"
            "5. Focus on the most important information\n"
            "6. Ensure the summary is self-contained and understandable"
        )

        # 여러 few-shot 예제를 하나의 문자열로 결합
        few_shot_examples = ""
        if isinstance(sample_dialogues, list) and isinstance(sample_summaries, list):
            for d, s in zip(sample_dialogues, sample_summaries):
                few_shot_examples += f"대화:\n{d}\n요약:\n{s}\n\n"
        else:
            few_shot_examples = f"대화:\n{sample_dialogues}\n요약:\n{sample_summaries}\n\n"

        user_prompt = (
            "다음은 대화 요약의 예시입니다. 이를 참고하여 새로운 대화를 요약해주세요.\n\n"
            f"{few_shot_examples}\n"
            "이제 아래 대화를 위 예시들과 비슷한 스타일로 요약해주세요.\n"
            f"대화:\n{dialogue}\n\n"
            "요약:"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    else:
        system_prompt = (
            "You are a Korean dialogue summarization expert.\n"
            "Create a concise and coherent summary that captures the key points of the dialogue.\n"
            "The summary should be in Korean and maintain the original context."
        )
        user_prompt = f"대화:\n{dialogue}\n\n요약:"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    return messages
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

# Validation data의 대화를 요약하고, 점수를 측정합니다.
def validate(config, val_df, client, sample_dialogues, sample_summaries, num_samples=-1):
    val_samples = val_df[:num_samples] if num_samples > 0 else val_df
    print("======= val_samples.shape: ", val_samples.shape)
    
    scores = []
    raw_scores = []
    predictions = []
    # Rate limiting을 위한 변수들
    requests_per_minute = 50  # 분당 최대 요청 수
    start_time = time.time()
    request_count = 0
    
    for idx, row in tqdm(val_samples.iterrows(), total=len(val_samples)):
        request_count += 1
        if request_count >= requests_per_minute:
            elapsed = time.time() - start_time
            if elapsed < 60:
                sleep_time = 60 - elapsed + 5  # 5초 추가 여유
                print(f"\nWaiting {sleep_time:.1f} seconds to respect rate limit...")
                time.sleep(sleep_time)
            start_time = time.time()
            request_count = 0
            
        dialogue = row['dialogue']
        try:
            summary = summarization(config, dialogue, client, sample_dialogues, sample_summaries)
            results = compute_metrics(summary, row['summary'])
            avg_score = sum(results.values()) / len(results)
            
            scores.append(avg_score)
            raw_scores.append(results)
            
            # 예측 결과 저장
            predictions.append({
                'dialogue': dialogue,
                'prediction': summary,
                'gold_summary': row['summary'],
                'rouge1_f1': results['rouge-1'],
                'rouge2_f1': results['rouge-2'],
                'rougeL_f1': results['rouge-l'],
                'avg_score': avg_score
            })
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {str(e)}")
            continue
    
    # 예측 결과를 DataFrame으로 변환하여 CSV로 저장
    predictions_df = pd.DataFrame(predictions)
    os.makedirs(config.general.output_path, exist_ok=True)
    val_output_path = os.path.join(config.general.output_path, "validation_samples.csv")

    # 텍스트 길이를 제한 (예: 1000자로 잘라서 저장)
    predictions_df["dialogue"] = predictions_df["dialogue"].apply(lambda x: x[:1000])

    # wandb에 결과 로깅
    table = wandb.Table(dataframe=predictions_df)
    wandb.log({"val_samples": table})
    predictions_df.to_csv(val_output_path, index=False)
    
    # wandb artifact 생성 및 로깅
    artifact = wandb.Artifact(
        name="validation_results", 
        type="dataset",
        description="Validation predictions and scores"
    )
    artifact.add_file(val_output_path)
    wandb.log_artifact(artifact)
    
    # wandb에 예측 결과 로깅
    for idx, row in predictions_df.iterrows():
        wandb.log({
            "val/dialogue": row['dialogue'],
            "val/prediction": row['prediction'],
            "val/gold_summary": row['gold_summary'],
            "val/rouge1_f1": row['rouge1_f1'],
            "val/rouge2_f1": row['rouge2_f1'],
            "val/rougeL_f1": row['rougeL_f1'],
            "val/avg_score": row['avg_score']
        })
    
    val_avg_score = sum(scores) / len(scores)
    print(f"\nValidation Average Score: {val_avg_score}")
    return raw_scores
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
    

@hydra.main(version_base="1.3", config_path="config", config_name="config")
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

    if cfg.custom_config.few_shot_selection == "max_len_summary":
        few_shot_samples = train_df.nlargest(cfg.custom_config.n_few_shot_samples, "summary_len")
    elif cfg.custom_config.few_shot_selection == "min_max_len_summary":
        few_shot_sample_min = train_df.nsmallest(cfg.custom_config.n_few_shot_samples, "summary_len")
        few_shot_sample_max = train_df.nlargest(cfg.custom_config.n_few_shot_samples, "summary_len")
        few_shot_samples = pd.concat([few_shot_sample_min, few_shot_sample_max])
    elif cfg.custom_config.few_shot_selection == "baseline":
        few_shot_samples = train_df.sample(cfg.custom_config.n_few_shot_samples, random_state=random_seed)
    else:
        raise ValueError(f"Invalid few_shot_selection value: {cfg.custom_config.few_shot_selection}")

    if cfg.custom_config.few_shot_selection == "baseline":
        sample_dialogue = few_shot_samples.iloc[0]['dialogue']
        sample_summary = few_shot_samples.iloc[0]['summary']
    else:
        sample_dialogue = few_shot_samples['dialogue'].tolist() #.iloc[0]['dialogue']
        sample_summary = few_shot_samples['summary'].tolist() #.iloc[0]['summary']

    print(f"Sample Dialogue1:\n{sample_dialogue}\n")
    print(f"Sample Summary1: {sample_summary}\n")

    client = OpenAI(
        api_key=UPSTAGE_API_KEY,
        base_url="https://api.upstage.ai/v1/solar"
    )

    scores =validate(cfg,val_df,client,sample_dialogue,sample_summary,n_val_samples)[0]
    print(scores)
    log_rouge(scores, wandb_logger)

    output = inference(cfg,client,test_df,sample_dialogue,sample_summary)
    wandb.finish()

if __name__ == "__main__":
    main()