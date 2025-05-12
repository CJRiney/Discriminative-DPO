import tqdm, json, time, sys, os, openai
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.data_util import CoTReasoning

device = 'cuda'

phi = '/mnt/data2/chris/research/bias-dpo/models/phi-3-mini-saved'

tokenizer = AutoTokenizer.from_pretrained(phi)

def phi_query(**kwargs):
    with openai.Client(api_key = "EMPTY", base_url="http://localhost:10000/v1/") as client:
        res = client.chat.completions.create(
            model = phi,
            **kwargs,
        )
    return res

with open("./test_prompts_generative.json", 'r', encoding='utf-8') as file:
    data = json.load(file)
    
results_path = "./stats/base_model_stats/base_model_accuracy_gen.json"
responses_path = "./stats/base_model_stats/base_model_responses_gen.json"
bool_array_path = "./stats/base_model_stats/base_model_correct_gen.json"
    
reasoner = CoTReasoning(None)
    
format = ("\n\nEnsure that your answer ends with \\underline{final answer}. "
        + "That is, if your final answer is 5, end your response with \\underline{5}.")
preface = ("In this task, I will give you a math question. "
        + "Your task is to reason and find the correct answer.\n\n")

bool_array = [False] * len(data)
corrects = 0
total = 0
responses = []
    
for i in tqdm.tqdm(range(len(data))):
    dp = data[i]
    answer = dp['correct']
    prompt = [{"role": "user", "content": preface + dp['prompt'] + format}]
    
    try: 
        response = phi_query(
            messages=prompt,
            temperature=0,
            max_tokens=1024
        ).choices[0].message.content
        
        print(response)
    
    except Exception as e: 
        print("Failed to retrieve phinetuned response due to the following error:")
        print(e)
        sys.exit()
        
    model_boxed = reasoner.extract_final(reasoner.extract_underline(response))
    if model_boxed =='[invalid]': continue
    
    total += 1
    
    if (model_boxed == answer): 
        bool_array[i] = True
        corrects += 1
        
    accuracy = {
        "accuracy": corrects / total,
        "correct": corrects,
        "total": total,
    }
    
    response = {
        "prompt": prompt,
        "response": response,
        "correct": answer,
        "model choice": model_boxed,
        "model correct": model_boxed == answer,
    }
    
    responses.append(response)
        
    with open(results_path, 'w', encoding='utf-8') as f: json.dump(accuracy, f, indent=4)
    with open(responses_path, 'w', encoding='utf-8') as f: json.dump(responses, f, indent=4)
    with open(bool_array_path, 'w', encoding='utf-8') as f: json.dump(bool_array, f, indent=4)