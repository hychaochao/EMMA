#!/bin/bash
OMP_NUM_THREADS=8 nohup srun \
 --partition=MoE \
 --mpi=pmi2 \
 --job-name=intern2 \
 -c 32 \
 -w SH-IDCA1404-10-140-54-102 \
 --ntasks-per-node=1 \
 --kill-on-bad-exit=1 \
 --quotatype=reserved \
python generate_response.py \
--subject 'Physics' 'Chemistry' \
--strategy 'Direct' \
--rerun \
--model_path '/mnt/petrelfs/share_data/quxiaoye/models/InternVL2-Llama3-76B' \
--output_path 'results/open-source/InternVL2_Direct.json' 1>logs/InternVL2_Direct.log 2>&1 &
# python generate_response.py \
# --subject 'Physics' 'Chemistry' \
# --strategy 'Direct' \
# --rerun \
# --model_path '/mnt/petrelfs/share_data/quxiaoye/models/Qwen2-VL-72B-Instruct' \
# --output_path 'results/open-source/Qwen_Direct.json' 1>logs/Qwen_Direct.log 2>&1 &
# python generate_response.py \
# --subject 'Physics' 'Chemistry' \
# --strategy 'Direct' \
# --rerun \
# --model_path '/mnt/petrelfs/share_data/quxiaoye/models/InternVL2_5-78B' \
# --output_path 'results/open-source/InternVL2_5_Direct.json' 1>logs/InternVL2_5_Direct.log 2>&1 &
# python generate_response.py \
# --subject 'Physics' 'Chemistry' \
# --strategy 'Direct' \
# --rerun \
# --model_path '/mnt/petrelfs/share_data/quxiaoye/models/llava-onevision-qwen2-72b-ov-hf' \
# --output_path 'results/open-source/Llava_Direct.json' 1>logs/Llava_Direct.log 2>&1 &































