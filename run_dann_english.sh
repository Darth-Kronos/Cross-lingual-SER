#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python main.py \
    --source_dataset "english_train" \
    --target_dataset "mandarin_test" \
    --perturb false; \
