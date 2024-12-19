import yaml
import json
import random
import re


def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            return yaml_dict
        except yaml.YAMLError as exc:
            print(exc)
            return None


def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True


def build_scoring_query(sample, config, total_num, select_num, seed):
    """Build the scoring query by combining the scoring_prompt and query. The <image_n> token is still there"""
    query = sample["query"]
    scoring_prompt = config["Scoring_prompt"]
    res_dict = {}

    # random select n responses from all responses
    assert select_num < total_num, f"select_num must be less than total_num"
    numbers = list(range(total_num))
    random.seed(seed)
    select_nums = random.sample(numbers, select_num)

    scoring_query = query
    for num in select_nums:
        if sample[f'response_{num}']:
            response = sample[f'response_{num}']
        else:
            response = "no response"
        scoring_query = scoring_query + f"\nResponse{{{num}}}:\nke{response}\n"
    scoring_query = scoring_query + scoring_prompt

    res_dict['scoring_query'] = scoring_query.strip()

    # append existing key and value in data
    res_dict.update(sample)
    return res_dict

def extract_score_list(text):

    pattern1 = r"inalscore(?:\{(\d+)\})\{[^\}]+\}\{(\d+)\}"
    pattern2 = r"inalscore(?:(\d+))\{[^\}]+\}\{(\d+)\}"

    # 创建列表并初始化为 None，长度为 16（因为最多有 16 个 response）
    scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for match in re.finditer(pattern2, text):
        response_number = int(match.group(1)) if match.group(1) else None  # 捕获 response_number
        score = int(match.group(2)) if match.group(2) else None  # 捕获 score
        if response_number:
            scores[response_number] = score  # 将 score 存储在对应的 response_number 位置

    for match in re.finditer(pattern1, text):
        response_number = int(match.group(1)) if match.group(1) else None  # 捕获 response_number
        score = int(match.group(2)) if match.group(2) else None  # 捕获 score
        if response_number is not None:
            scores[response_number] = score  # 将 score 存储在对应的 response_number 位置

    return scores
