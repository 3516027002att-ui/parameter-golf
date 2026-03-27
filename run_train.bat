@echo off
set MAX_WALLCLOCK_SECONDS=7200
set TRAIN_BATCH_TOKENS=65536
set RUN_ID=local_2h
set DATA_PATH=./data/datasets/fineweb10B_sp1024/
set TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
set VOCAB_SIZE=1024
D:\Repository\parameter-golf\.venv\Scripts\python.exe train_gpt.py
