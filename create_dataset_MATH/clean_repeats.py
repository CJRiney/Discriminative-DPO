import json

path = './MATH/sft_datasets/current_sft_train.jsonl'
prompts = {}

# Read the file and collect unique prompts with their responses
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        prompt = item['prompt']
        response = item['response']
        prompts[prompt] = response  # Ensure unique prompts, overwrite if duplicate

# Write back the unique prompts and responses to the file (clearing it first)
with open(path, 'w', encoding='utf-8') as f:
    for prompt, response in prompts.items():
        f.write(json.dumps({"prompt": prompt, "response": response}) + '\n')

print(f"Removed duplicates and saved unique prompts with responses back to {path}")

