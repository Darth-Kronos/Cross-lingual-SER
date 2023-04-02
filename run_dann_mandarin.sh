#!/bin/bash
CUDA_VISIBLE_DEVICES=6 python main.py \
    --source_dataset "mandarin_train" \
    --target_dataset "english_test" \
    --perturb false; \