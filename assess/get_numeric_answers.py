import tqdm, json, time, sys, os, openai
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.data_util import CoTReasoning
from tools.data_util import load_gsm_data

reasoner = CoTReasoning(None)

with open('./test_prompts.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
train_data, test_data = load_gsm_data()

data_set = set()
for dp in data:
    data_set.add(dp['prompt'].split('\n')[0][len('Question: '):])

new_data = []
for dp in train_data:
    if dp['question'] in data_set:
        new_data.append({
            'question': dp['question'],
            'answer': reasoner.extract_final(dp['answer'])
        })
        
with open('./test_prompts_generative.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=4)