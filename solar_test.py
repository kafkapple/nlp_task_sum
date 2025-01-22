import pandas as pd
import os
import time
from tqdm import tqdm
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.
from openai import OpenAI # openai==1.2.0


# 모델 성능에 대한 평가 지표를 정의합니다. 본 대회에서는 ROUGE 점수를 통해 모델의 성능을 평가합니다.
rouge = Rouge()


def compute_metrics(pred, gold):
    results = rouge.get_scores(pred, gold, avg=True)
    result = {key: value["f"] for key, value in results.items()}
    return result

# Dialogue를 입력으로 받아, Solar Chat API에 보낼 Prompt를 생성하는 함수를 정의합니다.
def build_prompt(dialogue):
    system_prompt = "You are an expert in the field of dialogue summarization. Please summarize the following dialogue."

    user_prompt = f"Dialogue:\n{dialogue}\n\nSummary:\n"
    
    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

def summarization(dialogue):
    summary = client.chat.completions.create(
        model="solar-1-mini-chat",
        messages=build_prompt(dialogue),
        temperature=0.1,
        top_p=0.1, #0.3
    )

    return summary.choices[0].message.content

# Train data 중 처음 3개의 대화를 요약합니다.
def test_on_train_data(num_samples=3):
    for idx, row in train_df[:num_samples].iterrows():
        dialogue = row['dialogue']
        summary = summarization(dialogue)
        print(f"Dialogue:\n{dialogue}\n")
        print(f"Pred Summary: {summary}\n")
        print(f"Gold Summary: {row['summary']}\n")
        print("=="*50)



# Validation data의 대화를 요약하고, 점수를 측정합니다.
def validate(num_samples=-1):
    val_samples = val_df[:num_samples] if num_samples > 0 else val_df
    
    scores = []
    raw_scores = []
    for idx, row in tqdm(val_samples.iterrows(), total=len(val_samples)):
        dialogue = row['dialogue']
        summary = summarization(dialogue)
        results = compute_metrics(summary, row['summary'])
        avg_score = sum(results.values()) / len(results)
        
        scores.append(avg_score)
        raw_scores.append(results)
    val_avg_score = sum(scores) / len(scores)

    print(f"Validation Average Score: {val_avg_score}")
    return raw_scores



def inference():
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

    summary = []
    start_time = time.time()
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        dialogue = row['dialogue']
        summary.append(summarization(dialogue))
        
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
    
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    output.to_csv(os.path.join(RESULT_PATH, "output_solar.csv"), index=False)

    return output


# Validation data의 대화를 요약하고, 점수를 측정합니다.
def calc_train_score(num_samples=-1):
    train_samples = train_df[:num_samples] if num_samples > 0 else train_df
    
    scores = []
    raw_scores = []
    for idx, row in tqdm(train_samples.iterrows(), total=len(train_samples)):
        dialogue = row['dialogue']
        summary = summarization(dialogue)
        results = compute_metrics(summary, row['summary'])
        avg_score = sum(results.values()) / len(results)
        
        scores.append(avg_score)
        raw_scores.append(results)
        for key, value in results.items():
            train_samples[f'{key}_score'] = value
        
    train_avg_score = sum(scores) / len(scores)

    print(f"Train Average Score: {train_avg_score}")
    return raw_scores

# Few-shot sample을 다른 방식으로 사용하여 prompt를 생성합니다.
def build_prompt(dialogue):
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
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": few_shot_user_prompt_1},
        {"role": "assistant", "content": few_shot_assistant_prompt_1},
        {"role": "user", "content": user_prompt},
    ]

output = inference()




random_seed = 2001
# Few-shot prompt를 생성하기 위해, train data의 일부를 사용합니다.
few_shot_samples = train_df.sample(1, random_state=random_seed)

sample_dialogue1 = few_shot_samples.iloc[0]['dialogue']
sample_summary1 = few_shot_samples.iloc[0]['summary']

print(f"Sample Dialogue1:\n{sample_dialogue1}\n")
print(f"Sample Summary1: {sample_summary1}\n")

train_df.head()

scores = validate(100)


UPSTAGE_API_KEY = ""
client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)

stream = client.chat.completions.create(
    model="solar-1-mini-chat",
    messages=[
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    stream=True,
)
 
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
 
# Use with stream=False
# print(stream.choices[0].message.content)

# 데이터 경로를 지정해줍니다.
DATA_PATH = "../data/"
RESULT_PATH = "./prediction/"

# train data의 구조와 내용을 확인합니다.
train_df = pd.read_csv(os.path.join(DATA_PATH,'train.csv'))
train_df.tail()

# validation data의 구조와 내용을 확인합니다.
val_df = pd.read_csv(os.path.join(DATA_PATH,'dev.csv'))
val_df.tail()
