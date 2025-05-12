import json

prompts, chosens, rejecteds = [], [], []

with open('local_datasets/dataset-QnA.json', 'r', encoding='utf-8') as file:
    dataset = json.load(file)

dicts = []

for item in dataset:
    new_item = {}
    new_item['prompt'] = item['prompt']
    new_item['chosen'] = item['chosen']
    new_item['rejected'] = item['rejected']
    dicts.append(new_item)

new_path = 'local_datasets/dataset-QnA.jsonl'

with open(new_path, 'w') as file:
    for item in dicts:
        file.write(json.dumps(item) + '\n')
