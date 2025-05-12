import json, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.MATH_util import MATHReasoning

util = MATHReasoning()

data_path = './MATH/dpo_datasets/dpo_train.jsonl'
target_path = './MATH/dpo_datasets/new_dataset.jsonl'

with open(data_path, 'r') as jsonl_file:
    data = []
    for line in jsonl_file:
        data.append(json.loads(line.strip()))
        
new_data =[]
for dp in data:
    prompt = dp['prompt']
    
    ans_a_start = prompt.find('Answer A:\n')
    ans_b_start = prompt.find('Answer B:\n')
    ans_c_start = prompt.find('Answer C:\n')
    
    ans_a = prompt[ans_a_start:ans_b_start]
    ans_b = prompt[ans_b_start:ans_c_start]
    ans_c = prompt[ans_c_start:]
    
    try:
        ans_a_boxed = util.last_boxed_only_string(ans_a)
        ans_a_final = util.remove_boxed(ans_a_boxed)
        ans_a_final = util.normalize(ans_a_final)
        
    except: continue
    
    try:
        ans_b_boxed = util.last_boxed_only_string(ans_b)
        ans_b_final = util.remove_boxed(ans_b_boxed)
        ans_b_final = util.normalize(ans_b_final)
        
    except: continue
    
    try:
        ans_c_boxed = util.last_boxed_only_string(ans_c)
        ans_c_final = util.remove_boxed(ans_c_boxed)
        ans_c_final = util.normalize(ans_c_final)
        
    except: continue

    ans_a_choices = sorted(ans_a_final.split(','))
    ans_b_choices = sorted(ans_b_final.split(','))
    ans_c_choices = sorted(ans_c_final.split(','))
    
    if (ans_a_choices == ans_b_choices): continue
    if (ans_b_choices == ans_c_choices): continue
    if (ans_c_choices == ans_a_choices): continue
    
    new_data.append(dp)
    
with open(target_path, 'w', encoding='utf-8') as f: 
    for dp in new_data:
        f.write(json.dumps(dp) + '\n')