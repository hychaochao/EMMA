python generate_response_ntimes.py \
--subject 'Math' \
--strategy 'CoT' \
--api_key '' \
--model 'chatgpt-4o-latest' \
--n_times 4 \
--output_path 'results/test-time-compute/GPT_Math_16.json' 1>logs/chatgpt-4o-latest_Math_16.log 2>&1 &