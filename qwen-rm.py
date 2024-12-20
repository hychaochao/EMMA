import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import logging
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='results/test-time-compute/qwen-rm-scoring/chatgpt-4o-latest_Math_16_raw.json')
    parser.add_argument('--save_every', type=int, default=1, help='save every n problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer generation')
    # Remote model
    parser.add_argument('--model', type=str, default="chatgpt-4o-latest", help='llm engine',
                        choices=['chatgpt-4o-latest', 'claude-3-5-sonnet-latest', 'gemini-2.0-flash-exp'])
    parser.add_argument('--api_key', type=str, default='')
    # Local model
    parser.add_argument('--model_path', type=str, default='/mnt/petrelfs/share_data/quxiaoye/models/Qwen2.5-Math-RM-72B')

    args = parser.parse_args()

    if os.path.exists(args.output_path):
        logging.info(f"Reading {args.output_path}")
        with open(args.output_path, 'r') as f:
            results = json.load(f)
    else:
        logging.error("{args.output_path} does not exist.}")


    full_pids = list(results.keys())
    skip_pids = []
    if not args.rerun and results:
        for pid, data in results.items():
            if 'score_list' in data and data['score_list'] is not None:
                skip_pids.append(pid)

        if len(skip_pids) > 0:
            logging.info(
                f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems...")

    test_pids = [pid for pid in full_pids if pid not in skip_pids]

    # Load Model
    # If we were given a custom path, load that model, otherwise use a remote service model
    if args.model_path:
        logging.info(f"Loading model from {args.model_path}...")

        model = AutoModel.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    logging.info(f"Model loaded!")


    logging.info(f"Starting to generate.....")
    for idx, pid in enumerate(tqdm(test_pids)):

        sample = results[pid].copy()

        chat_list = []
        try:
            for i in range(16):
                if sample[f'response_{i}']:
                    chat = [
                        {"role": "system", "content": "You are a math expert"},
                        {"role": "user", "content": sample['query']},
                        {"role": "assistant", "content": sample[f'response_{i}']}
                    ]
                else:
                    chat = [
                        {"role": "system", "content": "You are a math expert"},
                        {"role": "user", "content": sample['query']},
                        {"role": "assistant", "content": "no reponse"}
                    ]
                chat_list.append(chat)
            
            score_list = []
            for chat in chat_list:
                conversation_str = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=False
                )
                input_ids = tokenizer.encode(
                    conversation_str,
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(model.device)

                outputs = model(input_ids=input_ids)
                score_list.append(outputs[0].item())
            
            results[pid] = sample
            results[pid]['score_list'] = score_list

        except Exception as e:
            logging.error(f"Error in generating answer for {pid}")
            logging.error(e)
            results[pid] = sample
            results[pid]['error'] = str(e)

        if idx == 2 or (idx % args.save_every == 0 and idx > 0) or idx == len(results) - 1:
            try:
                with open(args.output_path, 'w') as f:
                    f.write(json.dumps(results, indent=2))
                logging.info(f"Save results to {args.output_path}")
            except Exception as e:
                logging.info(f"Error in saving {args.output_path}")
                logging.info(e)
    
    with open(args.output_path, 'w') as f:
        f.write(json.dumps(results, indent=2))
    logging.info(f"Save results to {args.output_path}")

    logging.info("End Generation......")


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]"
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()












