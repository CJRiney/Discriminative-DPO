import json, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.data_util import CoTReasoning

reasoner = CoTReasoning(None)

dposet_path = "./datasets/shuffled_train_dataset.jsonl"
testable_path = "./datasets/shuffled_train_testable.json"

prompts = []
chosens = []
test_sets = []
with open(dposet_path, 'r', encoding='utf-8') as f:
    for line in f:
        dp = json.loads(line)
        prompts.append(dp['prompt'])
        chosens.append(dp['chosen'])
        
for i in range(len(prompts)):
    prompt = prompts[i]
    chosen = chosens[i]
    
    start_idx = chosen.find("\\boxed{") + 7
    end_idx = start_idx + chosen[start_idx:].find('}')
    correct = reasoner.extract_final_mc3(chosen[start_idx:end_idx])
    
    if correct not in ['A', 'B', 'C']: continue
    
    test_set = {
        'prompt': prompt,
        'correct': correct
    }
    
    test_sets.append(test_set)
    
with open(testable_path, 'w', encoding='utf-8') as f: json.dump(test_sets, f, indent=4)
    
    
    
    
    
