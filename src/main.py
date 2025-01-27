# 데이터셋 준비
train_dataset = processor.prepare_dataset(
    "train",
    use_prompt=cfg.prompt.use_in_train  # 학습 시 프롬프트 사용 여부
)
val_dataset = processor.prepare_dataset(
    "validation",
    use_prompt=cfg.prompt.use_in_train  # 학습 시에도 동일한 설정 사용
)

# 추론 시
if cfg.custom_config.inference:
    inference = DialogueInference(cfg)
    inference.inference(
        cfg.general.data_path,
        use_prompt=cfg.prompt.use_in_inference  # 추론 시 프롬프트 사용 여부
    ) 