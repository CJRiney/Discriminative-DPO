# This program creates the multiple choice sets for the selected category, which can be modified on line 31.

import tqdm, openai, random, json, sys, os
from transformers import AutoTokenizer
from datasets import load_from_disk
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.MATH_util import MATHReasoning

@dataclass
class Answer:
    answers: list[str]
    count: int

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
letters =  ['A', 'B', 'C']

category = 'geometry'

results_path = f'./MATH/mc_sets/{category}.json'
few_shot_sets = open('./txts/few_shot.txt', 'r', encoding='utf-8').read().split('\n\n\n')

few_shot = []
for example in few_shot_sets:
    problem, solution = example.split('\n# Start of solution #\n', 1) 
    few_shot.append({ "role": "user", "content": problem })
    few_shot.append({ "role": "assistant", "content": solution })
    
request_boxed = "\nEnsure that your solution ends with \\boxed{answer}. For example, if your final answer is 5, your solution should end with \\boxed{5}."

test_data = load_from_disk(f'./MATH/docs/{category}')
target_attempts = 15
max_correct_rate = .3
mc_sets = []

for dp in tqdm.tqdm(test_data):   
    problem = dp['problem']
    solution = dp['solution']
    answer = util.normalize(dp['answer'])
    attempts = 0
    
    inst = [{ "role": "user", "content": problem + request_boxed }]
        
    answer_info = {}
    
    for _ in range(target_attempts):
        # Query the model
        response = model_query(
            messages = few_shot + inst,
            temperature=0.5,
            max_tokens=1024
        ).choices[0].message.content
        
        # Try to normalize the answer. Continue if unable.
        try:
            boxed = util.last_boxed_only_string(response)
            final = util.remove_boxed(boxed)
            final = util.normalize(final)
            attempts += 1
        except: continue
        
        # Check if the final response has been recorded.
        # If it has, add the response to the list of responses and add 1 to its count.
        # If it hasn't, then initialize a new Answer in the map with that response and count 1.
        if final in answer_info: 
            answer_info[final].answers.append(response)
            answer_info[final].count += 1
            
        else: answer_info[final] = Answer([response], 1)
        
    # If the model could not generate the correct answer,
    # then it is not expected to be able to select the correct answer.
    # If the model reaches the correct answer over 30% of the time,
    # then the task becomes generative, and this question cannot be used.
    # In either case, skip.
    if (answer not in answer_info 
        or answer_info[answer].count > attempts * max_correct_rate): continue
    
    # Extract the correct response and then delete it from the map
    correct_response = answer_info[answer].answers[0]
    del answer_info[answer]
    
    # First, get all the incorrect answers by extracting the keys.
    incorrect_answers = list(answer_info.keys())
    
    # If there are no incorrect answers due to excess skips
    if (len(incorrect_answers) == 0 or 
        (len(incorrect_answers) == 1 
        and answer_info[incorrect_answers[0]].count == 1)): continue
    
    # Then, if there are more than 1 incorrect answers, 
    # extract the first 2 different incorrect answers.
    # If there is only 1 incorrect answer,
    # just take 2 different responses with the same incorrect answer.
    if len(incorrect_answers) > 1: incorrect_responses = [answer_info[incorrect_answers[0]].answers[0], 
                                                          answer_info[incorrect_answers[1]].answers[0]] 
    else: incorrect_responses = [answer_info[incorrect_answers[0]].answers[0], 
                                 answer_info[incorrect_answers[0]].answers[1]]
    
    # Pick a random letter for the correct option, 
    # then pick different letters for the incorrect options.
    correct_letter = random.choice(letters)
    incorrect_letters = [letter for letter in letters if letter != correct_letter]
    
    # First add the correct letter mapped to the correct option,
    # then add the incorrect letters mapped to the incorrect options.
    mc_map = { correct_letter: correct_response }
    mc_map.update(dict(zip(incorrect_letters, incorrect_responses)))
    
    # Create the options from the multiple choice mappings
    options = (
        "Answer A:\n" + mc_map['A'] + '\n\n'
        + "Answer B:\n" + mc_map['B'] + '\n\n'
        + "Answer C:\n" + mc_map['C'] + '\n\n'
    )
    
    # Create the full multiple choice problem-options question
    mc_question = (problem + '\n\n' + options)
    
    mc_set = {
        "problem": problem,
        "mc question": mc_question,
        "answer": answer,
        "correct response": correct_response,
        "incorrect responses": incorrect_responses,
        "correct letter": correct_letter
    }
    
    mc_sets.append(mc_set)
    
    with open(results_path, 'w', encoding='utf-8') as f: json.dump(mc_sets, f, indent=4)
    

    
    
    
    
    
    
    
    
    
    
        
        