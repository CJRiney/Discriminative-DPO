import torch, json
import lightning as L
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Phi3ForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from trl import DPOConfig, DPOTrainer
from datasets import Dataset

device = 'cuda'
cache_dir = "./phi-3-mini"
max_seq_length = 2048 # Supports automatic RoPE Scaling, so choose any number.

with open('./hgf-dataset-QnA.json', 'r', encoding='utf-8') as file:
    train_dataset = Dataset.from_dict(json.load(file))

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    device_map=device, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True, 
    cache_dir="./phi-3-mini"
)

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    cache_dir=cache_dir
)

tokenizer.add_special_tokens({"pad_token":"<pad>"})

print(tokenizer.pad_token)
print(tokenizer.pad_token_id)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, r=8, 
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules = { 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj' },
)

model = get_peft_model(model, peft_config)

training_args = DPOConfig(
    output_dir="./output",
    beta=0.1,
    gradient_checkpointing=True,
    max_prompt_length=2048
)

dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()