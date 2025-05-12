import json
from pathlib import Path
from datasets import Dataset

target_dir = './MATH/used_problems/mc_questions.json'
main_dir = Path('./MATH/mc_sets')
datasets = [d for d in main_dir.iterdir()]

problems = []
for dataset in datasets:
    
    with open(dataset, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for dp in data:
        problems.append(dp['problem'])
        

with open(target_dir, 'w', encoding='utf-8') as f: json.dump(problems, f, indent=4)