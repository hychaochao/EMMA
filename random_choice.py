import argparse
import json
import os
import logging
from tqdm import tqdm
import random

from datasets import load_dataset, concatenate_datasets



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='mm-reasoning/EMMA')
    #parser.add_argument('--subject', nargs='+', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output_path', type=str, default='results/random_choice.json')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    results = {}

    # Load Dataset
    logging.info(f"Loading dataset {args.dataset_name}")
    subject = ['Math', 'Physics', 'Chemistry', 'Coding']
    sub_dataset_list = []
    for subj in subject:
        sub_dataset = load_dataset(args.dataset_name, subj, split=args.split)
        sub_dataset_list.append(sub_dataset)
    dataset = concatenate_datasets(sub_dataset_list)

    random.seed(args.seed)
    logging.info(f"Starting to generate.....")
    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        pid = sample['pid']
        if sample['type'].lower() == 'multiple choice':
            problem: dict = sample.copy()
            for i in range(1, 6):
                problem.pop('image_' + str(i))
            results[pid] = problem
            option_list = []
            for idx, option in enumerate(problem['options']):
                option_list.append(chr(65 + idx))
            results[pid]['response'] = random.sample(option_list, 1)[0]


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












