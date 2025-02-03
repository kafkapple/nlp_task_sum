# Dialogue Summarization Project

## Overview
- **Purpose**: Summarize Korean dialogues and optionally classify emotions.
- **Models**: 
  - BART (`gogamza/kobart-summarization`) for Seq2Seq Summarization
  - BERT (`klue/roberta-small`) for Classification
  - Llama (`MLP-KTLim/llama-3-Korean-Bllossom-8B`) for Causal LM
  - Solar API (Upstage) for Summarization (no fine-tuning)

## Project Structure
```bash
Project_Root/
├─ configs/
├─ src/
├─ data/  # ignored by git
├─ models/  # ignored by git
├─ outputs/  # ignored by git
├─ environment.yml
├─ .gitignore
├─ .env  # store API tokens
└─ README.md


Setup
```bash
conda env create -f environment.yml
conda activate dialogue_summary_env
```

Usage
Edit .env with your tokens:
```makefile
HUGGINGFACE_TOKEN=<your_token>
UPSTAGE_API_KEY=<your_upstage_key>
```

Download data (auto-downloaded if not present).
Run:
```bash
cd src
python main.py
```
Or override config:
```bash
python main.py model=bert train=finetune
```

References
- PyTorch Lightning
- Hugging Face Transformers
- Hydra Docs
- WandB Docs
