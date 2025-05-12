# Call this after 0shot_chosen.py to fill in the rejected responses

import json, tqdm, openai, sys, os
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.data_util import CoTReasoning

device='cuda'
model_id = '/mnt/data2/chris/research/bias-dpo/models/phi-3-mini-saved'
tokenizer = AutoTokenizer.from_pretrained(model_id)

def model_query(**kwargs):
    with openai.Client(api_key = "EMPTY",base_url="http://localhost:10001/v1/") as client:
        res = client.chat.completions.create(
            model = model_id,
            **kwargs,
        )
    return res

reasoner = CoTReasoning(None)

results_path = './datasets/dpo_synth-chosen_no-exp_filled1.json'
unfilled_path = './datasets/dpo_synth-chosen_no-exp.json'

with open(unfilled_path, 'r', encoding='utf-8') as f: data = json.load(f)

preface = ("In this task, I will give you a math question followed by 3 answers to the math question. "
        + "Your task is to identify which of the 3 answers are correct and explain why the answer you choose is correct. "
        + "Ensure that your answer ends with \\underline{correct option}. "
        + "That is, end your response with \\underline{A}, \\underline{B}, or \\underline{C}.\n\n")

results = []
for dp in tqdm.tqdm(data[:int(len(data) * .2)]):
    if dp['rejected'] != '': continue
    
    correct_option = dp['correct option']
    prompt = dp['prompt']
    inst = [{ "role": "user", "content": preface + prompt }]
    
    for _ in range(10):
        response = model_query(
            messages = inst,
            temperature=0.5,
            max_tokens=1024
        ).choices[0].message.content
        
        start_of_ans_idx = response.find('\\underline{') + 11
        end_of_ans_idx   = start_of_ans_idx + response[start_of_ans_idx:].find('}')
        try: model_boxed = reasoner.extract_final_mc3(response[start_of_ans_idx:end_of_ans_idx])
        except: continue
        
        if (model_boxed != correct_option): break
    
    if (model_boxed == correct_option): response = ''
    
    result = {
        'prompt': dp['prompt'],
        'chosen': dp['chosen'],
        'rejected': response,
        'correct option': correct_option
    }
    
    results.append(result)
    
    with open(results_path, 'w', encoding='utf-8') as f: json.dump(results, f, indent=4)