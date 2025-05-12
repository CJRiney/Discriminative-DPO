from datasets import load_dataset
from typing import List, Dict
import json

HUMAN_EVAL_STRINGS_OK = ["return x + y", "return len(string)", "return n**2", "return " ".join(strings)"]

def extract_docstring(prompt: str) -> str:
    if '"""' in prompt:
        if prompt.count('"""') == 2:
            return prompt.split('"""')[1].strip()
        elif prompt.count('"""') == 4:
            return prompt.split('"""')[3].strip()
        else:
            raise ValueError()
    elif "'''" in prompt:
        assert prompt.count("'''") == 2
        return prompt.split("'''")[1].strip()
    else:
        raise ValueError()
    
def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
    
def human_eval_docstrings() -> List[str]:
    ds = load_dataset("openai_humaneval", split="test")
    docstrings = [extract_docstring(v["prompt"]) for v in ds]
    return docstrings

def load_dataset_column(dataset: str, column: str, split: str, name=None) -> List[str]:
    ds = load_dataset(dataset, split=split, name=name)
    res = [sample[column].strip() for sample in ds]
    # Only return non-empty strings
    return [sample for sample in res if len(sample) > 0]

FILTER_OUT = {
    "human_eval_docstrings": human_eval_docstrings(),
    "human_eval_solutions": [
        s
        for s in load_dataset_column("openai_humaneval", "canonical_solution", "test")
        if s not in HUMAN_EVAL_STRINGS_OK
    ],
}

data = []
for _, substrings in FILTER_OUT.items():
    for substring in substrings:
        data.append(normalize_whitespace(substring.lower()))
        
with open('./unwanted_substrings.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)
                