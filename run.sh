#!/bin/bash
OMP_NUM_THREADS=8 nohup srun \
 --partition=MoE \
 --mpi=pmi2 \
 --job-name=hyz \
 -c 32 \
 -w SH-IDCA1404-10-140-54-89 \
 --ntasks-per-node=1 \
 --kill-on-bad-exit=1 \
 --quotatype=reserved \
python generate_response.py \
--subject 'Chemistry' 'Math' \
--strategy 'CoT' \
--model_path '/mnt/petrelfs/share_data/quxiaoye/models/InternVL2_5-78B' \
--output_path 'results/InternVL2_5_Math_Chemistry_CoT.json' 1>logs/InternVL2_5_Math_Chemistry_CoT.log 2>&1 &

# python generate_response.py \
# --subject 'Chemistry' 'Math' \
# --strategy 'CoT' \
# --model_path '/mnt/petrelfs/share_data/quxiaoye/models/llava-onevision-qwen2-72b-ov-hf' \
# --output_path 'results/Llava_Math_Chemistry_CoT.json' 1>logs/Llava_Math_Chemistry_CoT.log 2>&1 &

# python generate_response.py \
# --subject 'Chemistry' 'Math' \
# --strategy 'CoT' \
# --model_path '/mnt/petrelfs/share_data/quxiaoye/models/Qwen2-VL-72B-Instruct' \
# --output_path 'results/Qwen_Math_Chemistry_CoT.json' 1>logs/Qwen_Math_Chemistry_CoT.log 2>&1 &