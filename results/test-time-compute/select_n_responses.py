import json
import random
import os

def select_n_responses_with_score(input_path, output_path, n):

    with open(input_path, 'r') as f:
        data = json.load(f)

    seed = 42
    new_data = {}
    for pid, sample in data.items():
        random.seed(seed)
        nums = random.sample(range(16), n)
        new_data[pid] = sample.copy()
        for i in range(16):
            new_data[pid].pop(f'response_{i}')
        new_data[pid]['score_list'] = []
        for i in range(n):
            new_data[pid][f'response_{i}'] = sample[f'response_{nums[i]}']
            new_data[pid]['score_list'].append(sample['score_list'][nums[i]])

        score_list = new_data[pid]['score_list']

        max_score = max(score_list)
        max_indices = [i for i, score in enumerate(score_list) if score == max_score]

        # if there are many max scores, choose one randomly
        max_index = random.choice(max_indices)
        new_data[pid]['best_response'] = new_data[pid][f'response_{max_index}']

    with open(output_path, 'w') as f:
        f.write(json.dumps(new_data, indent=2))


def select_n_responses(input_path, output_path, total_n, select_n):
    with open(input_path, 'r') as f:
        data = json.load(f)

    seed = 42
    new_data = {}
    for pid, sample in data.items():
        random.seed(seed)
        nums = random.sample(range(total_n), select_n)
        new_data[pid] = sample.copy()
        for i in range(16):
            if f'response_{i}' in sample:
                new_data[pid].pop(f'response_{i}')
        for i in range(select_n):
            new_data[pid][f'response_{i}'] = sample[f'response_{nums[i]}']

    with open(output_path, 'w') as f:
        f.write(json.dumps(new_data, indent=2))

if __name__ == '__main__':
    for root, dirs, files in os.walk('raw_results'):
        for file in files:
            if 'Math' in file:
                select_n_responses(os.path.join(root, file), 'pass@1/'+file, 16, 1)
            else:
                select_n_responses(os.path.join(root, file), 'pass@1/'+file, 8, 1)


