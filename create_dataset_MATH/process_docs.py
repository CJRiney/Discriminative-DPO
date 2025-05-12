import json, os, sys
from pathlib import Path
from datasets import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.MATH_util import MATHReasoning

util = MATHReasoning()
    
target_dirs = ['algebra']
main_dir = Path('./MATH/train')
datasets = [d for d in main_dir.iterdir() if d.is_dir() and d.name in target_dirs]

for dataset in datasets:
    category = Path(dataset).name
    problems, solutions = [], []
    
    for data_file in dataset.iterdir():
        if data_file.suffix != '.json':
            continue
        
        with data_file.open('r', encoding='utf-8') as f:
            dp = json.load(f)
            
        problem = dp['problem']
        solution = dp['solution']
        
        problems.append(problem)
        solutions.append(solution)
    
    dataset = Dataset.from_dict({
            "problem": problems,
            "solution": solutions
        })
    
    processed_docs = util.process_docs(dataset)
    processed_docs.save_to_disk(f'./MATH/docs/{category}')

    
