#!/bin/bash
export CUDA_HOME=/mnt/petrelfs/share/cuda-11.8
export PATH=/mnt/petrelfs/share/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.8/lib64:$LD_LIBRARY_PATH
OMP_NUM_THREADS=8 nohup srun \
 --partition=MoE \
 --mpi=pmi2 \
 --job-name=Coding \
 -c 32 \
 -w SH-IDCA1404-10-140-54-89 \
 --ntasks-per-node=1 \
 --kill-on-bad-exit=1 \
 --quotatype=reserved \
python qwen-rm.py >logs/qwen_rm_gpt.log 2>&1 &
# python scoring.py \
# --subject 'Physics' \
# --total_num 8 \
# --select_num 8 \
# --model_path '/mnt/petrelfs/share_data/quxiaoye/models/InternVL2_5-78B' \
# --output_path 'results/test-time-compute/internvl-best-of-8/InternVL2_5_Physics_8.json' 1>logs/InternVL2_5_Physics_bo8.log 2>&1 &


# python generate_response.py \
# --subject 'Physics' 'Chemistry' \
# --strategy 'Direct' \
# --rerun \
# --model_path '/mnt/petrelfs/share_data/quxiaoye/models/InternVL2-Llama3-76B' \
# --output_path 'results/open-source/InternVL2_Direct.json' 1>logs/InternVL2_Direct.log 2>&1 &
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
# --model_path '/mnt/petrelfs/share_data/quxiaoye/models/llava-onevision-qwen2-72b-ov-hf' \
# --output_path 'results/open-source/Llava_Direct.json' 1>logs/Llava_Direct.log 2>&1 &































