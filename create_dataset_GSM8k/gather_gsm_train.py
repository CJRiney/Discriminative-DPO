import json, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.data_util import load_gsm_data

train_data = []
train, test = load_gsm_data()

for dp in train:
    new_dp = {
        'prompt': dp['question'],
        'response': dp['answer']
    }
    
    train_data.append(new_dp)
    
with open('./datasets/gsm8k_generation.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4)
    
with open('./datasets/gsm8k_generation.jsonl', 'w', encoding='utf-8') as f:
    for dp in train_data:
        f.write(json.dumps(dp) + '\n')
    
    