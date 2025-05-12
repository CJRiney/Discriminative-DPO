import json

data_path = './MATH/dpo_datasets/dpo_numina.json'
train_path = './MATH/dpo_datasets/dpo_numina_train.jsonl'
validation_path = './MATH/dpo_datasets/dpo_numina_validation.jsonl'
test_path = './MATH/dpo_datasets/dpo_numina_test.jsonl'
train_split = 1
test_split = 0

with open(data_path, 'r', encoding='utf-8') as f: 
    data = json.load(f)

with open(train_path, 'w', encoding='utf-8') as f: 
    for dp in data[:int(len(data) * train_split)]:
        f.write(json.dumps(dp) + '\n')
        
with open(validation_path, 'w', encoding='utf-8') as f: 
    for dp in data[int(len(data) * train_split) : int(len(data) * (train_split + test_split))]:
        f.write(json.dumps(dp) + '\n')
        
with open(test_path, 'w', encoding='utf-8') as f: 
    for dp in data[int(len(data) * (train_split + test_split)):]:
        f.write(json.dumps(dp) + '\n')

