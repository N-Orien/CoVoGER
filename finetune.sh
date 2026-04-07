#!/usr/bin/bash

MODEL_DIR=$1
DATA_DIR=$2
OUTPUT_DIR=$3

litgpt finetune_lora $MODEL_DIR \
  --data JSON \
  --data.json_path $DATA_DIR \
  --out_dir $OUTPUT_DIR \
  --train.epochs=1 \
  --train.save_interval=1000 \
  --eval.interval=100000 \
  --train.global_batch_size 64 \
  --train.micro_batch_size 8 \
  --train.max_seq_length 1024 \
  --precision bf16-true \
  --devices 1 &
