import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
import logging
from tqdm import tqdm
from datasets import load_dataset
from scoring_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='mm-reasoning/EMMA-test100')
    parser.add_argument('--subject', type=str, default='Math')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--config_path', type=str, default="configs/scoring.yaml")
    parser.add_argument('--output_path', type=str, default='results/test-time-compute/internvl-best-of-4/InternVL2_5_Math_16.json')
    parser.add_argument('--save_every', type=int, default=1, help='save every n problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer generation')
    parser.add_argument('--total_num', type=int, default=16, help='pass@n')
    parser.add_argument('--select_num', type=int, default=4, help='pass@n')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # Remote model
    parser.add_argument('--model', type=str, default="chatgpt-4o-latest", help='llm engine',
                        choices=['chatgpt-4o-latest', 'claude-3-5-sonnet-latest', 'gemini-2.0-flash-exp'])
    parser.add_argument('--api_key', type=str, default='')
    # Local model
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.7)

    args = parser.parse_args()

    if os.path.exists(args.output_path):
        logging.info(f"Reading {args.output_path}")
        with open(args.output_path, 'r') as f:
            results = json.load(f)
    else:
        logging.error("{args.output_path} does not exist.}")

    logging.info(f"Loading dataset {args.dataset_name}, subject: {args.subject}")
    dataset = load_dataset(args.dataset_name, args.subject, split=args.split)

    # Load Config
    logging.info(f"Loading config")
    config = load_yaml(args.config_path)

    # Load Model
    # If we were given a custom path, load that model, otherwise use a remote service model
    if args.model_path:
        logging.info(f"Loading local model {args.model_path}")
        if 'llava' in args.model_path.lower():
            from models import llava
            model = llava.Llava_Model(args.model_path, temperature=args.temperature, max_tokens=args.max_tokens)

        if 'qwen2-vl' in args.model_path.lower():
            from models import qwen
            model = qwen.Qwen_Model(args.model_path, temperature=args.temperature, max_tokens=args.max_tokens)

        if 'internvl' in args.model_path.lower():
            from models import internvl
            model = internvl.Internvl_Model(args.model_path, temperature=args.temperature, max_tokens=args.max_tokens)

    else:
        logging.info(f"Loading {args.model}")

        if 'gpt' in args.model.lower():
            from openai import OpenAI
            from models import gpt
            client = OpenAI(api_key=args.api_key)
            model = gpt.GPT_Model(client, args.model, temperature=args.temperature, max_tokens=args.max_tokens)

        elif 'claude' in args.model.lower():
            from anthropic import Anthropic
            from models import claude
            client = Anthropic(api_key=args.api_key)
            model = claude.Claude_Model(client, args.model, temperature=args.temperature, max_tokens=args.max_tokens)

        elif 'gemini' in args.model.lower():
            from openai import OpenAI
            from models import gpt
            client = OpenAI(
                api_key=args.api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            model = gpt.GPT_Model(client, args.model, temperature=args.temperature, max_tokens=args.max_tokens)

    logging.info(f"Model loaded!")

    skip_pids = []
    if not args.rerun and results:
        for pid, data in results.items():
            if 'scoring_response' in data and verify_response(data['scoring_response']):
                skip_pids.append(pid)

        if len(skip_pids) > 0:
            logging.info(
                f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems...")

    logging.info(f"Starting to generate.....")
    for idx, entry in enumerate(tqdm(dataset)):
        pid = entry['pid']
        if skip_pids and pid in skip_pids:
            continue

        sample = results[pid].copy()
        for i in range(1, 6):
            sample[f'image_{i}'] = entry[f'image_{i}']

        sample = build_scoring_query(sample, config, args.total_num, args.select_num, args.seed)
        problem: dict = sample.copy()
        for i in range(1, 6):
            problem.pop(f'image_{i}')

        try:
            response = model.get_response(sample)
            results[pid] = problem
            if response:
                results[pid]['scoring_response'] = response
                score_list = extract_score_list(response)
                logging.info(f"Scoring list:{score_list}")
                max_score = max(score_list)
                max_indices = [i for i, score in enumerate(score_list) if score == max_score]

                # if there are many max scores, choose one randomly
                max_index = random.choice(max_indices)
                results[pid]['best_response'] = problem[f'response_{max_index}']
                results[pid]['score_list'] = score_list

        except Exception as e:
            logging.error(f"Error in generating answer for {pid}")
            logging.error(e)
            results[pid] = problem
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












