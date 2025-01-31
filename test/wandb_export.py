# import wandb
# import pandas as pd

# # WandB 프로젝트 정보 입력
# project = "dialogue-summary"
# entity = "ailab_upstage_fastcampus"

# # WandB API로 데이터 가져오기
# api = wandb.Api()
# runs = api.runs(f"{entity}/{project}")

# # 필요한 데이터 추출
# data = []
# for run in runs:
#     data.append({**run.config, **run.summary, "name": run.name})

# # DataFrame 변환 후 Excel로 저장
# df = pd.DataFrame(data)
# df.to_excel("wandb_experiments.xlsx", index=False)
# print("WandB 실험 데이터가 Excel로 저장됨.")


import wandb
import pandas as pd


project = "dialogue-summary"
entity = "ailab_upstage_fastcampus"
# API로 데이터 가져오기
api = wandb.Api()
runs = api.runs(f"{entity}/{project}")

# 데이터 변환 (summary 안전 변환)
data = []
for run in runs:
    row = {
        "Run ID": run.id,
        "Name": run.name,
        "Created": str(run.created_at),  # 날짜를 문자열로 변환
    }
    # summary 값 변환 (get() 사용)
    for key, value in run.summary.items():
        row[key] = value if not isinstance(value, dict) else str(value)
    # config 값도 추가
    for key, value in run.config.items():
        row[key] = value if not isinstance(value, dict) else str(value)

    data.append(row)

# DataFrame 생성
df = pd.DataFrame(data)
# CSV 저장
df.to_csv("wandb_experiments.csv", index=False)
print("CSV 파일로 저장 완료.")
df.to_excel("wandb_experiments.xlsx", index=False, errors="coerce")
print("Excel 저장 완료 ✅")
