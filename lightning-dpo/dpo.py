import torch, datasets, random, yaml, math, copy
from typing import List, Dict

from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import TQDMProgressBar
from transformers import LlamaTokenizer, GemmaTokenizer, AutoTokenizer, AutoModelForCausalLM
from trl import setup_chat_format

from src.data_collator import DataCollatorWithPaddingSFT
from src.model import Phi3Mini
from src.my_strategy import MyDeepSpeedStrategy

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device='cuda'
config_path = 'config/base.yaml'
cache_dir = "./phi-3-mini"
output_dir = "./saved_models/output"
num_gpus=8

def main():
    with open(config_path, 'r', encoding = 'utf-8') as f:
        training_config = yaml.load(f, Loader=yaml.FullLoader)

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        cache_dir=cache_dir
    )
    tokenizer.add_special_tokens({"pad_token":"<pad>"}) 

    class MyTQDMBar(TQDMProgressBar):
        def __init__(self, refresh_rate: int = 1, process_position: int = 0):
            super().__init__(refresh_rate, process_position)
        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch and trainer.is_global_zero:
                print()
            super().on_train_epoch_start(trainer, pl_module)

    def tokenize(text):
        return tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=2048,
                add_special_tokens = True,
            )

    def preprocess_fn(examples: Dict[str, List]):
        processed_ids = {
            'chosen_ids': [],
            'rejected_ids': [],
            'c_start_idx': [],
            'r_start_idx': []
        }

        prompts = examples['prompt']
        chosens = examples['chosen']
        rejects = examples['rejected']

        resp_start_token = tokenize('<|assistant|>')['input_ids'][0]

        for i in range(len(prompts)):
            user_message = {'role': 'user', 'content': prompts[i]}
            chosen_resp = {'role': 'assistant', 'content': chosens[i]}
            rejected_resp = {'role': 'assistant', 'content': rejects[i]}

            chosen_example = [user_message, chosen_resp]
            rejected_example = [user_message, rejected_resp]

            ct_chosen = tokenizer.apply_chat_template(chosen_example, tokenize=False)
            ct_reject = tokenizer.apply_chat_template(rejected_example, tokenize=False)

            chosen_ids = tokenize(ct_chosen)
            rejected_ids = tokenize(ct_reject)

            c_start_idx = chosen_ids['input_ids'].index(resp_start_token)
            r_start_idx = rejected_ids['input_ids'].index(resp_start_token)

            processed_ids['chosen_ids'].append(chosen_ids['input_ids'])
            processed_ids['rejected_ids'].append(rejected_ids['input_ids'])
            processed_ids['c_start_idx'].append(c_start_idx)
            processed_ids['r_start_idx'].append(r_start_idx)
        
        return processed_ids

    remove_columns = ['prompt', 'chosen', 'rejected']
    train_data = datasets.load_dataset('json', data_files="local_datasets/train-dataset.jsonl", split='train')
    validation_data = datasets.load_dataset('json', data_files="local_datasets/validation-dataset.jsonl", split='train')
    train_data = train_data.map(preprocess_fn, batched=True, load_from_cache_file=True, remove_columns=remove_columns)
    validation_data = validation_data.map(preprocess_fn, batched=True, load_from_cache_file=True, remove_columns=remove_columns)

    example_data = []
    for i in range(len(train_data['chosen_ids'])):
        example_data.append({
            'chosen_ids': train_data['chosen_ids'][i],
            'rejected_ids': train_data['rejected_ids'][i],
            'c_start_idx': train_data['c_start_idx'][i],
            'r_start_idx': train_data['r_start_idx'][i]
        })

    collator = DataCollatorWithPaddingSFT(tokenizer=tokenizer, padding='longest')

    train_dataloader = DataLoader(
        train_data, shuffle=True, collate_fn=collator, batch_size=training_config['per_device_train_batch_size'],
        prefetch_factor = 10, num_workers = 10
    )
    eval_dataloader = DataLoader(
        validation_data, shuffle=False, collate_fn=collator, batch_size=training_config['per_device_eval_batch_size'],
        prefetch_factor = 10, num_workers = 10
    )

    model = Phi3Mini(tokenizer=tokenizer)

    devices = 1
    accelerator = 'gpu'
    strategy = MyDeepSpeedStrategy(stage=3)

    checkpoint_callback = ModelCheckpoint(dirpath=output_dir, save_top_k=5, monitor="valid_loss", every_n_epochs = 1, save_weights_only=True, save_last = True)
    all_callbacks = [checkpoint_callback]
    val_check_interval = training_config['checkpointing_steps']
    mylogger=loggers.WandbLogger(name = 'Phi3Mini-finetune', project = 'DPO', save_dir = './')

    trainer = L.Trainer(
        devices=[0],
        accelerator=accelerator,
        precision='bf16-true',
        strategy=strategy,
        logger=mylogger,
        val_check_interval = val_check_interval,
        max_epochs=training_config['num_train_epochs'],
        accumulate_grad_batches=training_config['gradient_accumulation_steps'],
        callbacks=all_callbacks,
        limit_val_batches=1.0,
        gradient_clip_val=training_config['max_grad_norm'],
    )

    trainer.fit(model, train_dataloader, eval_dataloader)
        

if __name__ == '__main__':
    main()




