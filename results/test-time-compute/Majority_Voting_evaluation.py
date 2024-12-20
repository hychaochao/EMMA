import argparse
import json
import logging
import os
from tqdm import tqdm
import re
from latex2sympy2 import latex2sympy
import re
from sympy import simplify
from collections import Counter
import random


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract_full_boxed_content(s):
    """
    Extract the full content inside \boxed{}, handling nested braces {{}} properly.
    """
    results = []

    i = 0
    while i < len(s):
        if s[i:i + 7] == r'\boxed{':
            brace_stack = []
            start = i + 7
            i = start

            while i < len(s):
                if s[i] == '{':
                    brace_stack.append(i)
                elif s[i] == '}':
                    if brace_stack:
                        brace_stack.pop()
                    else:
                        results.append(s[start:i])
                        break
                i += 1
        i += 1

    return results


def is_equal(md_ans, gt_ans):

    md_ans = md_ans.lower()
    gt_ans = gt_ans.lower()

    if md_ans.strip() == gt_ans.strip():
        return True

    # For Math
    try:
        # Parse LaTeX expressions into sympy and compare numerical values
        md_sympy = latex2sympy(md_ans)
        gt_sympy = latex2sympy(gt_ans)

        # Compare evaluated results, rounded to 2 decimal places
        if round(float(md_sympy.evalf()), 2) == round(float(gt_sympy.evalf()), 2):
            return True

        # Additionally, compare simplified symbolic expressions
        if simplify(md_sympy - gt_sympy) == 0:
            return True
    except Exception:
        pass  # Ignore parsing errors or evaluation failures

    return False


score_demo_prompt = """Please read the following example. Then determine whether the response is correct and type it 
at the end of the prompt. It is worth noting that the final answer in the response is usually in \\boxed{}, 
You only need to compare the final answer in the response with the answer, without considering the logical 
correctness of the response itself.

Response: The correct answer is:\n\nA

Answer: A

Correct_or_not: Correct

Response: The correct option is:\n\n\\[\n\\boxed{E}\n\\]

Answer: C

Correct_or_not: Incorrect
"""


def fast_extract_answer(response) :
    response = response.strip()
    # Direct Strategy Multi-Choice
    # A / A:
    for ch in 'ABCDEFGH':
        if response.upper() == ch or response.startswith(f'{ch}:'):
            return ch

    # Direct Strategy Open-ended
    # 1
    if is_number(response):
        return response

    # CoT strategy
    if 'boxed{' in response:
        try:
            model_answers = extract_full_boxed_content(response)
            if model_answers:
                # for coding
                # \\boxed{\\text{}}
                try:
                    text_content = re.findall(r'\\text{(.*?)}', model_answers[-1])
                    if text_content:
                        return text_content[-1].strip()
                except Exception:
                    pass
                return model_answers[-1].strip()
        except Exception:
            pass

    # for Coding
    # the correct answer is\n D.
    for flag in ['final answer is', 'correct answer is', 'answer should be', 'answer is']:
        if flag in response.lower():
            try:
                model_answer = response.lower().split(flag)[-1].strip()
                return model_answer.split('\n')[0].split('.')[0]
            except Exception:
                pass

    return response


def gen_true_false(answer_file, args):
    logging.info(f"Reading {answer_file}.....")
    label_list = args.response_label
    with open(answer_file, "r") as f:
        results = json.load(f)
    full_pids = list(results.keys())

    skip_pids = []
    for pid, problem in results.items():
        flag = problem.get('true_false')
        if flag is not None:
            skip_pids.append(problem['pid'])

    if args.rerun:
        test_pids = full_pids
    else:
        if len(skip_pids) > 0:
            logging.info(
                f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems..."
            )
        test_pids = [pid for pid in full_pids if pid not in skip_pids]

    logging.info(f"Number of test problems to run: {len(test_pids)}")

    for i, pid in enumerate(tqdm(test_pids)):
        problem = results[pid]
        results[pid]['extraction_list'] = []
        flag = False
        for label in label_list:
            if problem[label] is None:
                continue
            else:
                model_answer = fast_extract_answer(problem[label])
                results[pid]['extraction_list'].append(model_answer)

        counter = Counter(results[pid]['extraction_list'])

        max_count = max(counter.values())

        most_common_answers = [answer for answer, count in counter.items() if count == max_count]

        # 如果有多个相同次数的答案，随机选择一个
        random.seed(42)
        results[pid]['extraction'] = random.choice(most_common_answers)
        if is_equal(results[pid]['extraction'], results[pid]['answer']) or is_equal(results[pid]['extraction'], results[pid]['gt_content']):
            flag = True

        results[pid]['true_false'] = flag

        if (i % args.save_every == 0 and i > 0) or i == len(test_pids) - 1:
            with open(answer_file, "w") as f:
                f.write(json.dumps(results, indent=2))
            logging.info(f"Saved results to {answer_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_dir', type=str, default='/Users/chao/Desktop/Ashanghai/MultiBench/opensource/github/EMMA/results/test-time-compute/')
    parser.add_argument('--response_label', nargs='+', type=str, required=True)
    parser.add_argument('--rerun', action='store_true', help='rerun the answer extraction')
    parser.add_argument('--save_every', type=int, default=20, help='save every n problems')


    args = parser.parse_args()

    logging.info("Starting to extract answers.......")

    for root, dirs, files in os.walk(args.results_dir):
        for file in files:
            if file.endswith(".json") and not file.endswith("_result.json"):
                gen_true_false(os.path.join(root, file), args)

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