import json, tqdm, openai, sys, os
from transformers import AutoTokenizer
from datasets import load_from_disk
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.MATH_util import MATHReasoning

device='cuda'
model_id = '/mnt/data2/chris/research/bias-dpo/models/phi-3-mini-saved'
tokenizer = AutoTokenizer.from_pretrained(model_id)

def model_query(**kwargs):
    with openai.Client(api_key = "EMPTY",base_url="http://localhost:10000/v1/") as client:
        res = client.chat.completions.create(
            model = model_id,
            **kwargs,
        )
    return res

util = MATHReasoning()

category = 'algebra'

dir = Path(f'./MATH/phi-3-results/{category}')
dir.mkdir(parents=True, exist_ok=True)
responses_path = f'./MATH/phi-3-results/{category}/responses.json'
results_path = f'./MATH/phi-3-results/{category}/results.json'
few_shot_sets = open('./txts/few_shot.txt', 'r', encoding='utf-8').read().split('\n\n\n')

few_shot = []
for example in few_shot_sets:
    problem, solution = example.split('\n# Start of solution #\n', 1) 
    few_shot.append({ "role": "user", "content": problem })
    few_shot.append({ "role": "assistant", "content": solution })
    
request_boxed = "\nEnsure that your solution ends with \\boxed{answer}. For example, if your final answer is 5, your solution should end with \\boxed{5}."

test_data = load_from_disk(f'./MATH/docs/{category}')

responses = []
total, correct = 0, 0

for dp in tqdm.tqdm(test_data):   
    problem = dp['problem']
    solution = dp['solution']
    
    try: answer = util.normalize(dp['answer'])
    except: continue
    
    inst = [{ "role": "user", "content": problem + request_boxed }]
        
    response = model_query(
        messages = few_shot + inst,
        temperature=0,
        max_tokens=1024
    ).choices[0].message.content

    # Try to normalize answer. Skip if unable.
    try:
        boxed = util.last_boxed_only_string(response)
        final = util.remove_boxed(boxed)
        final = util.normalize(final)
    except: continue
    
    total += 1
    correct += util.is_equiv(final, answer)
    
    response = {
        "problem": problem,
        "response": response,
        "final": final,
        "correct": answer,
    }
    
    responses.append(response)
        
    result = {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
    }
    
    with open(responses_path, 'w', encoding='utf-8') as f: json.dump(responses, f, indent=4)
    with open(results_path, 'w', encoding='utf-8') as f: json.dump(result, f, indent=4)
        
            
            
            
            