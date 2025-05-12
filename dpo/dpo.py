'''
python tools/prepare.py
CUDA_VISIBLE_DEVICES=3,4,5,6 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/train_config.yaml ./dpo.py configs/dpo_config.yaml
'''
import logging
import random
import sys
from typing import List, Dict

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    # DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from peft import PeftConfig, PeftModel
from trl import DPOTrainer, DPOConfig
import datasets

logger = logging.getLogger(__name__)

def main():
    # These 2 lines are creating instances of dataclasses (model_args, data_args, training_args)
    # that parse through the dpo_config.yaml file and set any values in the dataclases 
    # matching the key values to whatever the key is set to in the yaml file
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", # Formatting stuff
        datefmt="%Y-%m-%d %H:%M:%S", # Formatting stuff
        handlers=[logging.StreamHandler(sys.stdout)], # This is saying put the output on the console
        # StreamHandler outputs logs to any streamlike object, and in this case we're using stdout, i.e. the console
    )
    log_level = training_args.get_process_log_level() # Get the log_level set in dpo_config that is now also set in training_args
    logger.setLevel(log_level) # Set the log level to that (usually INFO to skip overly-detailed debugging information)
    transformers.utils.logging.set_verbosity(log_level) # Set the log level of the transformers logger
    transformers.utils.logging.enable_default_handler() # Adds handler to transformers logger (not even needed since we made basicConfig, the root configuration)
    transformers.utils.logging.enable_explicit_format() # Allow transformers to use its own logging format instead of basicConfig

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    # raw_datasets = get_datasets(
    #     data_args,
    #     splits=data_args.dataset_splits,
    #     configs=data_args.dataset_configs,
    #     columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
    # )

    raw_datasets = datasets.load_from_disk(dataset_path = 'dataset/gsm8k_dpo')
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    
    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args) # There is already a default tokenizer for phi-3 so tokenizer is the default phi-3 tokenizer

    #####################
    # Apply chat template
    #####################
    column_names = list(raw_datasets["train"].features)
    print(column_names)
    print(raw_datasets['train'][0])
    sys.exit()
    raw_datasets = raw_datasets.map( # Go through each datapoint 1 by 1 and run apply_chat_template on it
        apply_chat_template,
        fn_kwargs={ # This fills the arguments of apply_chat_template
            "tokenizer": tokenizer,
            "task": "dpo",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers, # How many parallel processes should be used when applying apply_chat_template
        remove_columns=column_names, # If not removed, the unconverted prmopt, chosen, and rejected will still be there
        desc="Formatting comparisons with prompt template", # Description that will be displayed in progress bars
    )
    
    # Now, the prompt is converted to 'text_prompt': "<|system|>\n<|end|>\n<|user|>[prompt]<|end|>\n<|endoftext|>"
    # The chosen response is converted to 'text_chosen': "<|assistant|>\n[chosen response]<|end|>\n<|endoftext|>"
    # The rejected response is converted to 'text_rejected': "<|assistant|>\n[rejected response]<|end|>\n<|endoftext|>"

    ##########################
    # Decontaminate benchmarks
    ##########################
    num_raw_train_samples = len(raw_datasets["train"])
    raw_datasets = raw_datasets.filter( # Filter applies the filter function given (decontaminate_humaneval)
        decontaminate_humaneval,
        fn_kwargs={"text_column": "text_chosen"}, # Only clean the chosen data, because the dispreferred data needs to learn to be clean
        batched=True, # Do the filter in batches
        batch_size=10_000, # This is the size of the batch to be cleaned by decontaminate_humaneval
        num_proc=1,
        desc="Decontaminating HumanEval samples",
    )
    num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
    logger.info(
        f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
    )

    # Potential problem: This filter makes the chosen response whitespace only, getting rid of newlines and tabs. 
    # Since the math responses are format-heavy, this could be placing the focus of training on the change
    # in formatting instead of making the responses more correct. Also, we are using the human_eval dataset to filter
    # out data for some reason. Look at unwanted_substrings.json to see what we're trying to filter out. Makes no sense.

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    # for split in ["train", "test"]:
    for split in ["train", 'test']:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )
        
    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    torch_dtype = ( # We do this check because we can't do getattr(torch, None) or getattr(torch, "auto")
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args) # Use model_args to configure the floating point precision information
    # "Quantization is a technique to reduce the computational and memory costs of running inference by representing the weights and activations 
    # with low-precision data types like 8-bit integer (int8) instead of the usual 32-bit floating point (float32)." - Huggingface
    
    # By the way, in model_args, it is None

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
    target_modules = { 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj' }
    # microsoft/Phi-3-mini-4k-instruct is not an adapter model. This is skipped.
    if is_adapter_model(model, model_args.model_revision) is True:
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision, target_modules=target_modules)
        peft_config.lora_alpha = 64

        model_kwargs = dict(
            revision=model_args.base_model_revision, # This is the only change
            trust_remote_code=model_args.trust_remote_code,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
            cache_dir = './model_cache'
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            cache_dir = './model_cache',
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
        model_kwargs = None

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True: # It is
        ref_model = None
        ref_model_kwargs = None
    training_args.ref_model_init_kwargs = ref_model_kwargs
    # training_args.generate_during_eval = False
    # training_args.is_encoder_decoder = False
    # training_args.model_adapter_name = None
    # training_args.ref_adapter_name = None
    # training_args.reference_free = False
    # training_args.precompute_ref_log_probs = False
    # training_args.force_use_ref_model = False
    # training_args.max_target_length = None
    # training_args.label_pad_token_id = -100
    # training_args.disable_dropout = True
    # training_args.truncation_mode = "keep_end"
    # training_args.loss_type = "sigmoid"
    # training_args.label_smoothing = 0
    # training_args.f_divergence_type = 'reverse_kl'
    # training_args.f_alpha_divergence_coef = 1.0

    ###########################
    # Instantiate DPO trainer #
    ###########################
    trainer = DPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
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
    # For an even worse time, click into the DPOTrainer class as well
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
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
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()