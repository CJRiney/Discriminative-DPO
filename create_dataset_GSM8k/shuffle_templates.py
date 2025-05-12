import json, random, sys

with open('./datasets/prompts_no_exp.json', 'r', encoding='utf-8') as f: data = json.load(f)

def edit0(text):
    new_text = text.replace("Question: ", '')
    
    return new_text

def edit1(text):
    new_text = text + ("\n3 researchers offer their reasoning and answer in response to this question. "
                    + "Researcher A offers answer A, researcher B offers answer B, and researcher C offers answer C. "
                    + "Which answer is correct?\n\n")

    return new_text

def edit2(text):
    new_text = text + ("\nAnalyze these options and select the correct option.\n\n")
    
    return new_text

def edit3(text):
    new_text = text.replace("Answer A:\n", 'A:')
    new_text = new_text.replace("Answer B:\n", 'B:')
    new_text = new_text.replace("Answer C:\n", 'C:')
    
    return new_text

def edit4(text):
    new_text = text.replace("Answer A:\n", 'A) ')
    new_text = new_text.replace("Answer B:\n", 'B) ')
    new_text = new_text.replace("Answer C:\n", 'C) ')
    
    return new_text

def edit5(text):
    new_text = text.replace("Answer A:\n", 'A.')
    new_text = new_text.replace("Answer B:\n", 'B.')
    new_text = new_text.replace("Answer C:\n", 'C.')
    
    return new_text


results = []
for dp in data:
    new_prompt = dp['prompt'].split('\n\n')
    question = new_prompt[0]
    options = new_prompt[1]
    
    if random.choice([True, False]): question = edit0(question)
    
    if random.choice([True, False]):
        randint = random.randint(0, 1)
        edit_functions = [edit1, edit2]
        question = edit_functions[randint](question)
        
    else: question += '\n\n'
    
    randint = random.randint(0, 2)
    edit_functions = [edit3, edit4, edit5]
    options = edit_functions[randint](options)
    
    dp['prompt'] = question + options
    results.append(dp)
    
with open('./datasets/prompts_shuffled-templates_no-exp.json', 'w', encoding='utf-8') as f: json.dump(results, f, indent=4)