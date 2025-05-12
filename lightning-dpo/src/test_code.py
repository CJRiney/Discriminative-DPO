### Ensure processed_ids has correct start indices ###
    # processed_ids = preprocess_fn(train_data)

    # for i in range(len(processed_ids['prompt'])):
    #     c_start_idx = processed_ids['c_start_idx'][i]
    #     print('\n\nIDX:', i, '\n')
    #     print(tokenizer.decode(processed_ids['chosen'][i]['input_ids'][c_start_idx:]))

    # print('#### Combined and tokenized ####')
    # print(chosen_example)
    # print(tokenizer.encode)
    # print('\n\n')
    # print(user_message, chosen_resp)

### Ensure collator is masking correctly ###
    # collator = DataCollatorWithPaddingSFT(tokenizer=tokenizer, padding='longest')
    # c_mask = collator(example_data)['c_mask']
    # r_mask = collator(example_data)['r_mask']
    # for i in range(len(c_mask)):
    #     print(c_mask[i][example_data[i]['c_start_idx'] - 1:len(example_data[i]['chosen_ids']) + 2])
    #     print(r_mask[i][example_data[i]['r_start_idx'] - 1:len(example_data[i]['rejected_ids']) + 2])

# ref = AutoModelForCausalLM.from_pretrained(
    #     "microsoft/Phi-3-mini-4k-instruct", 
    #     device_map='cuda', 
    #     torch_dtype=torch.bfloat16, 
    #     trust_remote_code=True, 
    #     cache_dir="./phi-3-mini"
    # )
            
    # opt = copy.deepcopy(ref)

    # def shared_forward(inputs, splits = 'train'):
    #         chosen_ids = inputs['chosen_ids']
    #         rejected_ids = inputs['rejected_ids']

    #         ref_logits = ref(chosen_ids)
    #         opt_logits = opt(rejected_ids)

    #         return ref_logits.shape

    #         ref_probs = F.softmax(ref_logits)
    #         opt_probs = F.softmax(opt_logits)

    #         c_masked_ids = chosen_ids * inputs['c_mask']
    #         r_masked_ids = rejected_ids * inputs['r_mask']

    #         pi_opt_w = torch.sum(torch.log(ref_probs.gather(1, chosen_ids)))

    # print(shared_forward(input_ids))