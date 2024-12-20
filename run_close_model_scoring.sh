#!/bin/bash

# Subjects and strategy
SUBJECT="Physics" # You can use multiple subjects separated by spaces
STRATEGY="Direct" # CoT or Direct

# Remote proprietary model selection
MODEL="chatgpt-4o-latest" # Remote model name
API_KEY=""


# Default additional parameters
MAX_TOKENS=2048
TEMPERATURE=0
SAVE_EVERY=1
DATASET_NAME="mm-reasoning/EMMA-test100"
SPLIT="test"
CONFIG_PATH="configs/scoring.yaml"
TOTAL_NUM=8
SELECT_NUM=4
SEED=42
RERUN=""



# Replace spaces in subjects with underscores for file names
SUBJECT_FORMATTED=$(echo $SUBJECT | tr ' ' '_')

# Construct output and log file paths
OUTPUT_FILE="/Users/geniusgu/Code/Githubs/EMMA/results/test-time-compute/gpt-best-of-4/${MODEL}_${SUBJECT_FORMATTED}_${TOTAL_NUM}.json"
LOG_FILE="logs/${MODEL}_${SUBJECT_FORMATTED}_${TOTAL_NUM}.log"

# Print constructed file paths for debugging
echo "==============================================="
echo "🚀 Starting Script Execution"
echo "==============================================="
echo "📁 Output File Path: ${OUTPUT_FILE}"
echo "📝 Log File Path:    ${LOG_FILE}"
echo "-----------------------------------------------"
echo "🔧 Configuration Details:"
echo "   - Subjects:       ${SUBJECT}"
echo "   - Strategy:       ${STRATEGY}"
echo "   - Model:          ${MODEL}"
echo "   - Total Num:      ${TOTAL_NUM}"
echo "   - Select Num:     ${SELECT_NUM}"
echo "   - Seed:           ${SEED}"
echo "   - Rerun:          ${RERUN}"
echo "-----------------------------------------------"

# Run the script
python scoring.py  \
  --dataset_name $DATASET_NAME \
  --rerun \
  --subject $SUBJECT \
  --split $SPLIT \
  --config_path $CONFIG_PATH \
  --output_path $OUTPUT_FILE \
  --save_every $SAVE_EVERY \
  --total_num $TOTAL_NUM \
  --select_num $SELECT_NUM \
  --seed $SEED \
  --model $MODEL \
  --api_key $API_KEY \
  --model_path "" \
  --max_tokens $MAX_TOKENS \
  --temperature $TEMPERATURE \
  $RERUN_FLAG 1>$LOG_FILE 2>&1 &

# Completion message
echo "✅ Script launched successfully!"
echo "==============================================="
