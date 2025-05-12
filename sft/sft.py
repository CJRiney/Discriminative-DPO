'''
python3 prepare_sft.py
CUDA_VISIBLE_DEVICES=0,2,3,4 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/train_config.yaml ./sft.py configs/sft_config.yaml
'''

import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.log_config import logger

import logging
import sys

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, DefaultDataCollator, set_seed

from alignment import (
    get_peft_config,
    get_tokenizer,
    apply_chat_template,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
    ModelArguments,
    DataArguments,
    H4ArgumentParser
)
from transformers import Trainer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig
from peft import get_peft_model

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")
    
    tokenizer = get_tokenizer(model_args, data_args)
    
    # Check for last checkpoint
    last_checkpoint = None
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")
        
    set_seed(training_args.seed)
    
    raw_dataset = datasets.load_from_disk(dataset_path = 'dataset/data_sft')
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_dataset.items()]}"
    )
    
    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args) # There is already a default tokenizer for phi-3 so tokenizer is the default phi-3 tokenizer
    
    #####################
    # Apply chat template
    #####################
    column_names = list(raw_dataset["train"].features)
    sft_dataset = raw_dataset.map( # Go through each datapoint 1 by 1 and run apply_chat_template on it
        apply_chat_template,
        fn_kwargs={ # This fills the arguments of apply_chat_template
            "tokenizer": tokenizer,
            "task": "sft",
            "auto_insert_empty_system_msg": False,
        },
        num_proc=data_args.preprocessing_num_workers, # How many parallel processes should be used when applying apply_chat_template
        remove_columns=column_names, # If not removed, the unconverted columns will still be there
        desc="Formatting comparisons with prompt template", # Description that will be displayed in progress bars
    )
    
    column_names = list(sft_dataset["train"].features)

    # Apply the tokenizer to each datapoint in the dataset
    sft_dataset = sft_dataset.map(
        lambda examples: tokenizer(examples['text']),
        remove_columns=column_names,  # Removes original columns after tokenization
        batched=True  # Apply tokenization in batches for efficiency
    )
    
    instruction_template = "<|user|>"
    response_template = '<|assistant|>'
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, 
                                               response_template=response_template, 
                                               tokenizer=tokenizer, mlm=False)
    
    train_dataset = sft_dataset["train"]
    eval_dataset = sft_dataset["test"]
    
    torch_dtype = ( # We do this check because we can't do getattr(torch, None) or getattr(torch, "auto")
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    
    model_kwargs = dict(
        revision=model_args.model_revision, # What branch to load the model from, "main" by default, which is most up-to-date branch
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True, # We do have gradient checkpointing on so False
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config, 
        cache_dir = './model_cache'
    )

    model = model_args.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    model = get_peft_model(model, get_peft_config(model_args))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    
    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint) # Click into the train function for a bad time
    metrics = train_result.metrics
    metrics["train_samples"] = len(sft_dataset["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model('./output')
    logger.info(f"Model saved to {'./output'}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained('./output')

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(sft_dataset["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")
    
    
if __name__ == "__main__":
    main()