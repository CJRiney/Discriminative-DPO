import datasets
import json
import sys, os
from typing import List, Dict
from pathlib import Path

def convert_dataset_format(dataset: List[Dict]):
    converted_dataset = []
    for data_dict in dataset:
        prompt = data_dict['prompt']
        response = data_dict['response']

        formatted_data = {
            'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
        }
        converted_dataset.append(formatted_data)
    return converted_dataset


def save_to_disk(dataset_path: str):
    dataset = datasets.load_dataset('json', data_files = dataset_path)
    save_path = 'dataset/data_sft'
    dataset.save_to_disk(save_path)


def main():
    data_path = 'dataset/raw_data/train-dataset.jsonl'
    save_path = 'logs/formatted_data/converted_v1_train.jsonl'
    
    dir = Path(f'logs/formatted_data')
    dir.mkdir(parents=True, exist_ok=True)
    
    dataset = []
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    converted_dataset = convert_dataset_format(dataset)
    with open(save_path, 'w') as f:
        for line in converted_dataset:
            f.write(json.dumps(line) + '\n')


    data_path = 'dataset/raw_data/validation-dataset.jsonl'
    save_path = 'logs/formatted_data/converted_v1_eval.jsonl'
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
        'train': 'logs/formatted_data/converted_v1_train.jsonl',
        'test': 'logs/formatted_data/converted_v1_eval.jsonl',
    }
    save_to_disk(path_dict)

if __name__ == '__main__':
    main()