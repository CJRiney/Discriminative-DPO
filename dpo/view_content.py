import json

with open('./dataset/raw_gsm/train-dataset.jsonl') as file:
    for line in file:
        line = json.load(line.strip())
        print(line['prompt'])
        break