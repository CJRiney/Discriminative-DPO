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

def given(option):
    return f"Ensure that your answer ends with \\underline{{{option}}}"


reasoner = CoTReasoning(None)

dialogue = [{ "role": "user", "content": "In this task, I will give you a multiple choice math question, along with the correct option. "
        + "You will answer the question as if you are taking an exam, by stating which option is correct and explaining why it is correct. "
        + "Or, you can state which option is correct, and explain why the other options are incorrect. "
        + "End your response with \\underline{correct option}. That is, end your response with \\underline{A}, \\underline{B}, or \\underline{C}. "
        + "First, analyze the option, then output the correct boxed letter answer."
        + "\n\nDo you understand the task?" }]

dialogue.append(
        { "role": "assistant", "content": "Yes, I understand the task. "
        + "Please provide the math question along with the answer options and the correct answer, "
        + "and I will answer the question as if I am taking an exam, by stating which option is correct and explaining why it is correct, " 
        + "or stating which option is correct and explaining why the other options are incorrect. "
        + "I will end my response with \\underline{correct option}. That is, \\underline{A}, \\underline{B}, or \\underline{C}. "
        + "I will first analyze the option, then output the correct boxed letter answer."})

with open('./datasets/prompts_shuffled-templates_no-exp.json', 'r', encoding='utf-8') as f: data = json.load(f)

results = []
for dp in tqdm.tqdm(data):
    correct_option = dp['correct']
    prompt = dp['prompt'] + '\n\n' + 'Explain why ' + correct_option + ' is the correct option.' '\n' + given(correct_option)
    inst = [{ "role": "user", "content": prompt }]
    
    response = model_query(
        messages = dialogue + inst,
        temperature=0,
        max_tokens=1024
    ).choices[0].message.content
    
    start_of_ans_idx = response.find('\\underline{') + 11
    end_of_ans_idx   = start_of_ans_idx + response[start_of_ans_idx:].find('}')
    try: model_boxed = reasoner.extract_final_mc3(response[start_of_ans_idx:end_of_ans_idx])
    except: continue
    
    if (model_boxed != correct_option): continue
    
    result = {
        'prompt': dp['prompt'],
        'chosen': response,
        'rejected': '',
        'correct option': correct_option
    }
    
    results.append(result)
    
    with open('./datasets/dpo_synth-chosen_no-exp.json', 'w', encoding='utf-8') as f: json.dump(results, f, indent=4)