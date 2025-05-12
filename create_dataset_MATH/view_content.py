import json, os, sys
from pathlib import Path
from datasets import Dataset

# main_dir = Path('./MATH/mc_sets')
# datasets = [d for d in main_dir.iterdir()]

# datapoints = 0
# for dataset in datasets:
#     with open(dataset, 'r', encoding='utf-8') as f:
#         datapoints += len(json.load(f))
        
# print(datapoints)

# with open('./MATH/mc_sets/precalculus.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
    
# dp = data[2]
# print(dp['mc question'])
# print(dp['answer'])
# print(dp['correct letter'])

count = 0
with open('./MATH/dpo_datasets/dpo.json', 'r', encoding='utf-8') as f: data = json.load(f)
with open('./MATH/dpo_datasets/new_dataset.jsonl', 'r') as jsonl_file:
    data = []
    for line in jsonl_file:
        count += 1

print(len(data))
print(count)