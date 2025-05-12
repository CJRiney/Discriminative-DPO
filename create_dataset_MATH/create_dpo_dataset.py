import json, sys, os, time, tqdm
from pathlib import Path
from openai import AzureOpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.data_util import CoTReasoning

from dotenv import load_dotenv

load_dotenv()

gpt_api_key = my_secret = os.getenv("GPT_API_KEY")

client = AzureOpenAI(
        api_version="2024-02-01",
        azure_endpoint="https://gpt4-1106-ucnlp.openai.azure.com/",
        api_key=gpt_api_key,
    )

def gpt_complete(**kwargs):
    response = None
    while response == None:
        while True:
            try:
                client = AzureOpenAI(
                    api_version="2024-02-01",
                    azure_endpoint="https://gpt4-1106-ucnlp.openai.azure.com/",
                    api_key=gpt_api_key,
                )
                result = client.chat.completions.create(
                    model="GPT4-o",
                    **kwargs,
                )
                break
            except Exception as e:
                print(e)
                reason = e.body['code']
                if reason == 'content_filter':
                    return None
                time.sleep(3)
        response = result.choices[0].message.content
    return response

def given(option):
    return f"Ensure that your answer ends with \\underline{{{option}}}"

device='cuda'
reasoner = CoTReasoning(None)
target_path = './MATH/dpo_datasets/dpo_numina.json'

directory = Path(target_path).parent
directory.mkdir(parents=True, exist_ok=True)

main_dir = Path('./MATH/mc_sets')
datasets = [d for d in main_dir.iterdir() if d.name == 'numina.json']

### This section is for creating the chosen responses ###   

preface = ("In this task, I will give you a multiple choice math question, along with the correct option. "
        + "You will answer the question as if you are taking an exam, by stating which option is correct and explaining why it is correct. "
        + "Or, you can state which option is correct, and explain why the other options are incorrect. "
        + "End your response with \\underline{correct option}. That is, end your response with \\underline{A}, \\underline{B}, or \\underline{C}. "
        + "First, analyze the option, then output the correct boxed letter answer.\n\n")

dpo_dps = []
for dataset in tqdm.tqdm(datasets):
    with open(dataset, 'r', encoding='utf-8') as f: 
        data = json.load(f)
        
    for dp in tqdm.tqdm(data[244:]):
        mc_question = dp['mc question']
        correct_option = dp['correct letter']
        
        prompt = (preface + mc_question + '\n\n' 
                + 'Explain why ' + correct_option 
                + ' is the correct option.' 
                + '\n' + given(correct_option))
        
        response = gpt_complete(
            messages = [{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1024
        )
        
        final = reasoner.extract_formatted_ABC(response, '\\underline{', '}')
        
        if (final == None or final != correct_option): continue
        
        dpo_dp = {
            'prompt': mc_question,
            'chosen': response,
            'rejected': '',
        }
        
        dpo_dps.append(dpo_dp)
        
        with open(target_path, 'w', encoding='utf-8') as f: 
            json.dump(dpo_dps, f, indent=4)
            
### This section is for creating the rejected responses ###            

preface = ("In this task, I will give you a multiple choice math question, along with the correct option. "
        + "You will pick an incorrect option intentionally, and try to convince the reader that the option is correct. "
        + "Ensure that you do not mention that the option you are picking is incorrect. "
        + "End your response with \\underline{chosen option}. "
        + "That is, end your response with \\underline{A}, \\underline{B}, or \\underline{C}.\n\n")

with open(target_path, 'r', encoding='utf-8') as f: 
    data = json.load(f)
    
for i in tqdm.tqdm(range(len(data))):
    dp = data[i]
    mc_question = dp['prompt']
    chosen = dp['chosen']
    correct_option = reasoner.extract_formatted_ABC(chosen, '\\underline{', '}')
    
    prompt = (preface + mc_question + '\n\n' 
            + correct_option + ' is the correct option. ' 
            + 'Convince the reader that a different option is correct.')
    
    response = gpt_complete(
        messages = [{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=1024
    )
    
    final = reasoner.extract_formatted_ABC(response, '\\underline{', '}')
    
    if (final == None or final == correct_option): continue
    
    data[i] = {
        'prompt': mc_question,
        'chosen': chosen,
        'rejected': response,
    }
    
    with open(target_path, 'w', encoding='utf-8') as f: 
        json.dump(data, f, indent=4)
        
for dp in data:
    if dp['chosen'] == "" or dp['rejected'] == "":
        data.remove(dp)
        
with open('./MATH/dpo_datasets/dpo_numina_test.json', 'w', encoding='utf-8') as f: 
    json.dump(data, f, indent=4)
        
        
        
        
        

