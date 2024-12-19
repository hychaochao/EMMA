import json
import random

with open('InternVL2_5_Math_16_raw.json', 'r') as f:
    data = json.load(f)

seed = 42
new_data = {}
n = 1
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

with open('scored_results/InternVL2_5_Math_1.json', 'w') as f:
    f.write(json.dumps(new_data, indent=2))


