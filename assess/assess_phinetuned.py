import tqdm, json, time, sys, os, openai
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModelForCausalLM, LoraConfig
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.data_util import CoTReasoning

device = 'cuda'

model = 'phi-3-mc-and-gen'
mode = 'sft'

phi = '/mnt/data2/chris/research/bias-dpo/models/phi-3-mini-saved'
phinetuned = f'/mnt/data2/chris/research/bias-dpo/models/{model}'

lora_cfg = LoraConfig.from_pretrained(phinetuned)
tokenizer = AutoTokenizer.from_pretrained(phi)

phi_model = AutoModelForCausalLM.from_pretrained(phi, trust_remote_code=True)
phi_model.load_adapter(phinetuned, adapter_name=mode)
phi_model.active_adapter = mode
phi_mocdel = phi_model.to(device)

def phinetuned_query(input, max_tokens):
    prompt = tokenizer.apply_chat_template(
        input, tokenize=False,
        add_generation_prompt=True,
        )
    tokenized_prompt = tokenizer(prompt, return_tensors='pt').to(device)
    
    response = phi_model.generate(**tokenized_prompt, max_length=len(tokenized_prompt[0]) + max_tokens)
    message = tokenizer.decode(response[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
    
    return message

dir = Path(f'./stats/{model}')
dir.mkdir(parents=True, exist_ok=True)
results_path = f"./stats/{model}/{model}_mc_accuracy.json"
responses_path = f"./stats/{model}/{model}_mc_responses.json"

with open("./test_prompts.json", 'r', encoding='utf-8') as file:
    data = json.load(file)
    
with open("./stats/base_model_stats/base_model_correct_mc.json", 'r', encoding='utf-8') as file:
    base_model_correct = json.load(file)
    
reasoner = CoTReasoning(None)
    
format = ("\n\nEnsure that your answer ends with \\underline{correct option}. "
        + "That is, end your response with \\underline{A}, \\underline{B}, or \\underline{C}.")

preface = ("In this task, I will give you a math question followed by 3 answers to the math question. "
        + "Your task is to identify which of the 3 answers are correct and explain why the answer you choose is correct.\n\n")

corrects = 0
total = 0
response_sets = []

for i in tqdm.tqdm(range(len(data))):
    dp = data[i]
    answer = dp['correct']
    marked = False
    prompt = [{"role": "user", "content": preface + dp['prompt'] + format}]
    
    try: 
        response = phinetuned_query(
            prompt,
            max_tokens = 1024
        )
    
        print(response)
    
    except Exception as e: 
        print("Failed to retrieve phinetuned response due to the following error:")
        print(e)
        sys.exit()
        
    model_boxed = reasoner.extract_formatted_ABC(response, '\\underline{', '}')
    if model_boxed == '[invalid]': continue

    total += 1
    
    if (model_boxed == answer): 
        corrects += 1
        if not base_model_correct[i]: marked = True
        
    result = { 
        "accuracy": corrects / total,
        "correct": corrects,
        "total": total,
    }

    response_set = {
        "prompt": prompt,
        "response": response,
        "correct": answer,
        "model choice": model_boxed,
        "marked": marked,
        "model correct": model_boxed == answer,
        "base model correct": base_model_correct[i],
    }
    
    response_sets.append(response_set)
    
    with open(results_path, 'w', encoding='utf-8') as file: json.dump(result, file, indent=4)
    with open(responses_path, 'w', encoding='utf-8') as file: json.dump(response_sets, file, indent=4)