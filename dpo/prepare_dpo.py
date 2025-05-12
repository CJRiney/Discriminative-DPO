import datasets
import json
import os
from typing import List, Dict

def convert_dataset_format(dataset: List[Dict]):
    converted_dataset = []
    for data_dict in dataset:
        prompt = data_dict['prompt']
        chosen = data_dict['chosen']
        rejected = data_dict['rejected']

        formated_data = {
            'prompt': prompt,
            'chosen': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': chosen}],
            'rejected': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': rejected}]
        }
        converted_dataset.append(formated_data)
    return converted_dataset


def save_to_disk(dataset_path: str):
    dataset = datasets.load_dataset('json', data_files = dataset_path)
    save_path = 'dataset/gsm8k_dpo'
    dataset.save_to_disk(save_path)


def main():
    data_path = 'dataset/raw_gsm/train-dataset.jsonl'
    save_path = 'logs/gsm_data/converted_v1_train.jsonl'
    dataset = []
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    converted_dataset = convert_dataset_format(dataset)
    with open(save_path, 'w') as f:
        for line in converted_dataset:
            f.write(json.dumps(line) + '\n')


    data_path = 'dataset/raw_gsm/validation-dataset.jsonl'
    save_path = 'logs/gsm_data/converted_v1_eval.jsonl'
    dataset = []
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    converted_dataset = convert_dataset_format(dataset)
    with open(save_path, 'w') as f:
        for line in converted_dataset:
            f.write(json.dumps(line) + '\n')

    path_dict = {
        'train': 'logs/gsm_data/converted_v1_train.jsonl',
        'test': 'logs/gsm_data/converted_v1_eval.jsonl',
    }
    save_to_disk(path_dict)

if __name__ == '__main__':
    main()


