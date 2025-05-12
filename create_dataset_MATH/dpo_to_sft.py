import json

dpo_path = './MATH/dpo_datasets/current_dpo_train.jsonl'
sft_path = './MATH/sft_datasets/current_sft_train.jsonl'

# Open the input JSONL file and create the transformed output
with open(dpo_path, "r") as infile, open(sft_path, "w") as outfile:
    for line in infile:
        item = json.loads(line)
        # Transform the item
        transformed_item = {
            "prompt": item["prompt"],
            "response": item["chosen"]
        }
        # Write the transformed item to the output file
        outfile.write(json.dumps(transformed_item) + "\n")

print(f"Transformed dataset saved to {sft_path}")