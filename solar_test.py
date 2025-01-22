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


def init_logger(cfg: DictConfig):

    return WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name= f"{cfg.model.name}_{cfg.model.mode}_{cfg.general.timestamp}",
        config=OmegaConf.to_container(cfg, resolve=True),
        group=cfg.model.family,
        dir=cfg.general.output_path,
    )
load_dotenv()


def compute_metrics(pred, gold):
    # 모델 성능에 대한 평가 지표를 정의합니다. 본 대회에서는 ROUGE 점수를 통해 모델의 성능을 평가합니다.    
    rouge = Rouge()
    results = rouge.get_scores(pred, gold, avg=True)
    result = {key: value["f"] for key, value in results.items()}
    return result

# Dialogue를 입력으로 받아, Solar Chat API에 보낼 Prompt를 생성하는 함수를 정의합니다.
# def build_prompt(config,dialogue, sample_dialogue1, sample_summary1):
#     if config["custom_config"]["few_shot"]:
#         system_prompt = "You are a expert in the field of dialogue summarization, summarize the given dialogue in a concise manner. Follow the user's instruction carefully and provide a summary that is relevant to the dialogue."

#         few_shot_user_prompt_1 = (
#             "Following the instructions below, summarize the given document.\n"
#             "Instructions:\n"
#             "1. Read the provided sample dialogue and corresponding summary.\n"
#             "2. Read the dialogue carefully.\n"
#             "3. Following the sample's style of summary, provide a concise summary of the given dialogue. Be sure that the summary is simple but captures the essence of the dialogue.\n\n"
#             "Dialogue:\n"
#             f"{sample_dialogue1}\n\n"
#             "Summary:\n"
#         )
#         few_shot_assistant_prompt_1 = sample_summary1
        
#         user_prompt = (
#             "Dialogue:\n"
#             f"{dialogue}\n\n"
#             "Summary:\n"
#         )
#         messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": few_shot_user_prompt_1},
#         {"role": "assistant", "content": few_shot_assistant_prompt_1},
#         {"role": "user", "content": user_prompt},
#         ]


#     else:
#         system_prompt = "You are an expert in the field of dialogue summarization. Please summarize the following dialogue."
#         user_prompt = f"Dialogue:\n{dialogue}\n\nSummary:\n"
#         messages = [
#         {
#             "role": "system",
#             "content": system_prompt
#         },
#         {
#             "role": "user",
#             "content": user_prompt
#         }
#         ]
#     return messages
def build_prompt(config, dialogue, sample_dialogues, sample_summaries):
    """
    sample_dialogues: list of sample dialogues
    sample_summaries: list of corresponding summaries
    """
    if config["custom_config"]["few_shot"]:
        system_prompt = "You are a expert in the field of dialogue summarization, summarize the given dialogue in a concise manner. Follow the user's instruction carefully and provide a summary that is relevant to the dialogue."

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add few shot examples
        for dialogue_sample, summary_sample in zip(sample_dialogues, sample_summaries):
            few_shot_user_prompt = (
                "Following the instructions below, summarize the given document.\n"
                "Instructions:\n"
                "1. Read the provided sample dialogue and corresponding summary.\n"
                "2. Read the dialogue carefully.\n"
                "3. Following the sample's style of summary, provide a concise summary of the given dialogue. Be sure that the summary is simple but captures the essence of the dialogue.\n\n"
                "Dialogue:\n"
                f"{dialogue_sample}\n\n"
                "Summary:\n"
            )
            messages.extend([
                {"role": "user", "content": few_shot_user_prompt},
                {"role": "assistant", "content": summary_sample}
            ])
        
        # Add target dialogue
        user_prompt = (
            "Dialogue:\n"
            f"{dialogue}\n\n"
            "Summary:\n"
        )
        messages.append({"role": "user", "content": user_prompt})

    else:
        system_prompt = "You are an expert in the field of dialogue summarization. Please summarize the following dialogue."
        user_prompt = f"Dialogue:\n{dialogue}\n\nSummary:\n"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    return messages
def summarization(config, dialogue,client, sample_dialogue1, sample_summary1):
    messages = build_prompt(config, dialogue,sample_dialogue1,sample_summary1)
    max_retries = 3
    retry_delay = 60  # 1분 대기
    # summary = client.chat.completions.create(
    #     model="solar-1-mini-chat",
    #     messages=messages,
    #     temperature=config["custom_config"]["temperature"],
    #     top_p=config["custom_config"]["top_p"]
    # )
    # return summary.choices[0].message.content
    for attempt in range(max_retries):
        try:
            summary = client.chat.completions.create(
                model="solar-1-mini-chat",
                messages=messages,
                temperature=config["custom_config"]["temperature"],
                top_p=config["custom_config"]["top_p"]
            )
            return summary.choices[0].message.content
        except openai.RateLimitError:
            if attempt < max_retries - 1:  # 마지막 시도가 아니면
                print(f"\nRate limit reached. Waiting {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 대기 시간을 2배로 증가
            else:
                raise  # 최대 시도 횟수를 초과하면 예외를 발생시킴

# Validation data의 대화를 요약하고, 점수를 측정합니다.
def validate(config, val_df,client, sample_dialogues, sample_summaries, num_samples=-1):
    val_samples = val_df[:num_samples] if num_samples > 0 else val_df
    print("======= val_samples.shape: ", val_samples.shape)
    
    scores = []
    raw_scores = []
    # Rate limiting을 위한 변수들
    requests_per_minute = 50  # 분당 최대 요청 수
    start_time = time.time()
    request_count = 0
    for idx, row in tqdm(val_samples.iterrows(), total=len(val_samples)):

    #     dialogue = row['dialogue']
    #     summary = summarization(config, dialogue, client, sample_dialogue1, sample_summary1)
    #     results = compute_metrics(summary, row['summary'])
    #     avg_score = sum(results.values()) / len(results)
        
    #     scores.append(avg_score)
    #     raw_scores.append(results)
    # val_avg_score = sum(scores) / len(scores)

    # print(f"Validation Average Score: {val_avg_score}")
    # return raw_scores
     # Rate limiting 체크
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
        except Exception as e:
            print(f"\nError processing sample {idx}: {str(e)}")
            continue
    
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
    for idx, row in train_df[:num_samples].iterrows():
        dialogue = row['dialogue']
        summary = summarization(config, dialogue, client, sample_dialogue1, sample_summary1)
        print(f"Dialogue:\n{dialogue}\n")
        print(f"Pred Summary: {summary}\n")
        print(f"Gold Summary: {row['summary']}\n")
        print("=="*50)

def eda(df):
    df['dialogue_len'] = df.dialogue.str.len()
    try:
        df['summary_len'] = df.summary.str.len()
    except:
        print("Test data has no summary column")
    agg_df = df.agg(['mean','std','min','max'])
    print(agg_df)
    print(df.describe())

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
    else:
        few_shot_samples = train_df.sample(cfg.custom_config.n_few_shot_samples, random_state=random_seed)

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