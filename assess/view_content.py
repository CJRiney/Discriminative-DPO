import json

with open('./base_model_stats/base_model_correct_mc.json', 'r', encoding='utf-8') as f:
    base_model_corrects = json.load(f)
    
with open('./outputs/shuffled-templates_mc_responses.json', 'r', encoding='utf-8') as f:
    mc_responses = json.load(f)
    
with open('./outputs/shuffled-templates_mc_accuracy.json', 'r', encoding='utf-8') as f:
    mc_accuracy = json.load(f)
    
with open('./outputs/test_prompts.json', 'r', encoding='utf-8') as f:
    test_prompts = json.load(f)
    
new_results = []
correct_forced = 0
total_forced = 0
for i in range(len(test_prompts)):
    test_prompt_dp  = test_prompts[i]
    
    forced = test_prompt_dp['forced']
    
    if forced: total_forced += 1
    if mc_responses[i]['model correct'] and forced: correct_forced += 1
    
new_accuracy = {
    "accuracy": 0.535796766743649,
    "correct": 232,
    "total": 433,
    "forced accuracy": correct_forced / total_forced,
    "forced correct": correct_forced,
    "forced total": total_forced
}

with open('./outputs/shuffled-templates_mc_accuracy.json', 'w', encoding='utf-8') as f:
    json.dump(new_accuracy, f, indent=4)
    
    
    

