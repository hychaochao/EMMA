#!/bin/bash

# Subjects and strategy
SUBJECT="Coding" # You can use multiple subjects separated by spaces
STRATEGY="CoT" # CoT or Direct

# Remote proprietary model selection
MODEL="chatgpt-4o-latest" # Remote model name
API_KEY=""
N_TIMES=8
# Default additional parameters
MAX_TOKENS=2048
TEMPERATURE=0.7
SAVE_EVERY=1
SPLIT="test"
CONFIG_PATH="configs/gpt.yaml"

# Replace spaces in subjects with underscores for file names
SUBJECT_FORMATTED=$(echo $SUBJECT | tr ' ' '_')
# Construct output and log file paths
OUTPUT_FILE="results/test-time-compute/${MODEL}_${SUBJECT_FORMATTED}_${N_TIMES}.json"
LOG_FILE="logs/${MODEL}_${SUBJECT_FORMATTED}_${STRATEGY}_${N_TIMES}.log"

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
echo "-----------------------------------------------"

# Run the script
python generate_response_ntimes.py  \
  --n_times $N_TIMES \
  --subject $SUBJECT \
  --split $SPLIT \
  --strategy $STRATEGY \
  --output_path $OUTPUT_FILE \
  --model $MODEL \
  --api_key $API_KEY \
  --model_path "" \
  --config_path $CONFIG_PATH \
  --max_tokens $MAX_TOKENS \
  --temperature $TEMPERATURE \
  --save_every $SAVE_EVERY 1>$LOG_FILE 2>&1 &

# Completion message
echo "✅ Script launched successfully!"
echo "==============================================="
