import torch, json, tqdm, copy
from safetensors.torch import save_model
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Phi3ForCausalLM
from src.data_collator import DataCollatorWithPaddingSFT
from torch.nn import functional as F
from peft import LoraConfig, TaskType, get_peft_model

torch.random.manual_seed(0)

cache_dir = "./phi-3-mini"
device = 'cuda'
target_modules = { 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj' }
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, r=8, 
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules = target_modules,
)

model_ref = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    device_map=device, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True, 
    cache_dir="./phi-3-mini"
)

model_opt = copy.deepcopy(model_ref)
model_opt = get_peft_model(model_opt, peft_config)

model_ref.require_grad = False

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    cache_dir=cache_dir
)

tokenizer.add_special_tokens({"pad_token":"<pad>"}) 

with open('./dataset-QnA.json', 'r', encoding='utf-8') as file:
    train_data = json.load(file)[:100]

with open('./dataset-QnA.json', 'r', encoding='utf-8') as file:
    val_data = json.load(file)[100:]

# input_ids = tokenizer.encode(input, return_tensors="pt").to(device)

# outputs = model(input_ids)
# logits_per_position = outputs.logits
# print(logits_per_position)

learning_rate = 2e-5
num_epochs = 100
sample_size = 100
beta = .1

optimizer = torch.optim.AdamW(model_ref.parameters(), lr=learning_rate)

def train(model_opt):
    for i in tqdm.tqdm(range(len(train_data))):
        #### TRAINING ####
        ### Prepare necessary information ###

        i = 0

        dp = train_data[i]

        # Get the prompt, rejected response, and chosen response
        prompt = dp['prompt']
        rejected = dp['prompt'] + dp['rejected']
        chosen = dp['prompt'] + dp['chosen']

        # Get the input ID's for the prompt, rejected response, and chosen response
        print("#### TRAINING ####")
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        print(prompt_ids.shape)
        wrong_ids = tokenizer.encode(rejected, return_tensors='pt').to(device)
        print(wrong_ids.shape)
        right_ids = tokenizer.encode(chosen, return_tensors='pt').to(device)
        print(right_ids.shape)
        start_idx = len(prompt_ids[0]) # This is the starting point of the logits that we want to consider

        # Optimal and reference model's logits for wrong response and right response, including the prompt in context
        opt_logits_l = model_opt(wrong_ids).logits[0][start_idx:]
        ref_logits_l = model_ref(wrong_ids).logits[0][start_idx:]
        opt_logits_w = model_opt(right_ids).logits[0][start_idx:]
        ref_logits_w = model_ref(right_ids).logits[0][start_idx:]

        ### Begin constructing loss equation under Bradley-Terry model ###

        # First redefine wrong and right ID's to the ID's of the responses without the prompt
        wrong_ids = wrong_ids[0][start_idx:].unsqueeze(-1)
        right_ids = right_ids[0][start_idx:].unsqueeze(-1)

        # Calculate the mean probability of optimal and reference models generating rejected and chosen response
        pi_opt_l = torch.sum(torch.log(F.softmax(opt_logits_l, dim=1).gather(1, wrong_ids).squeeze()))
        pi_ref_l = torch.sum(torch.log(F.softmax(ref_logits_l, dim=1).gather(1, wrong_ids).squeeze()))
        pi_opt_w = torch.sum(torch.log(F.softmax(opt_logits_w, dim=1).gather(1, wrong_ids).squeeze()))
        pi_ref_w = torch.sum(torch.log(F.softmax(ref_logits_w, dim=1).gather(1, wrong_ids).squeeze()))

        # # Define the loss equation
        loss = -torch.log(torch.sigmoid(beta * (pi_opt_w - pi_ref_w)) - (beta * (pi_opt_l - pi_ref_l)))
        loss.backward()
        optimizer.step()
        
        # Track loss
        print("#### Training Loss ####")
        print(loss.item())

        #########################################################################################################

        #### VALIDATION ####
        ### Prepare necessary information ###

        dp = val_data[i % len(val_data)]

        # Get the prompt, rejected response, and chosen response
        prompt = dp['prompt']
        rejected = dp['prompt'] + dp['rejected']
        chosen = dp['prompt'] + dp['chosen']

        # Get the input ID's for the prompt, rejected response, and chosen response
        print("#### VALIDATION ####")
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        print(prompt_ids.size())
        wrong_ids = tokenizer.encode(rejected, return_tensors='pt').to(device)
        print(wrong_ids.size())
        right_ids = tokenizer.encode(chosen, return_tensors='pt').to(device)
        print(right_ids.size())
        start_idx = len(prompt_ids[0]) # This is the starting point of the logits that we want to consider

        with torch.no_grad():
            # Optimal and reference model's logits for wrong response and right response, including the prompt in context
            opt_logits_l = model_opt(wrong_ids).logits[0][start_idx:]
            ref_logits_l = model_ref(wrong_ids).logits[0][start_idx:]
            opt_logits_w = model_opt(right_ids).logits[0][start_idx:]
            ref_logits_w = model_ref(right_ids).logits[0][start_idx:]

            ### Begin constructing loss equation under Bradley-Terry model ###

            # First redefine wrong and right ID's to the ID's of the responses without the prompt
            wrong_ids = wrong_ids[0][start_idx:].unsqueeze(-1)
            right_ids = right_ids[0][start_idx:].unsqueeze(-1)

            # Calculate the mean probability of optimal and reference models generating rejected and chosen response
            pi_opt_l = torch.sum(torch.log(F.softmax(opt_logits_l, dim=1).gather(1, wrong_ids).squeeze()))
            pi_ref_l = torch.sum(torch.log(F.softmax(ref_logits_l, dim=1).gather(1, wrong_ids).squeeze()))
            pi_opt_w = torch.sum(torch.log(F.softmax(opt_logits_w, dim=1).gather(1, wrong_ids).squeeze()))
            pi_ref_w = torch.sum(torch.log(F.softmax(ref_logits_w, dim=1).gather(1, wrong_ids).squeeze()))

            # Define the loss equation
            loss = -torch.log(torch.sigmoid(beta * (pi_opt_w - pi_ref_w)) - (beta * (pi_opt_l - pi_ref_l)))

            # Track loss
            print("#### Validation Loss ####")
            print(loss.item())

train(model_ref)