import json

with open('./datasets/dpo_synth-chosen_no-exp.json', 'r', encoding='utf-8') as f: data = json.load(f)

print(len(data))

