import torch, copy
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import lightning as L
from torch.nn import CrossEntropyLoss, KLDivLoss
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

class Phi3Mini(L.LightningModule):
    def __init__(self, device='cuda', tokenizer = None):
        super().__init__()
        target_modules = { 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj' }
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, r=8, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules = target_modules,
        )

        self.ref = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            trust_remote_code=True, 
            cache_dir="./phi-3-mini"
        )
        
        self.opt = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            trust_remote_code=True, 
            cache_dir="./phi-3-mini"
        )
        
        self.opt = get_peft_model(self.opt, peft_config)
        self.ref.require_grad = False

        self.ref.resize_token_embeddings(len(tokenizer))
        self.opt.resize_token_embeddings(len(tokenizer))

    def shared_forward(self, inputs, batch_idx, splits = 'train'):
        beta = 0.1
        # batch size x max sequence
        chosen_ids = inputs['chosen_ids']
        rejected_ids = inputs['rejected_ids']

        print("### INPUT IDS SHAPE ###")
        print("Chosen ID's:", chosen_ids.shape)
        print("Rejected ID's:",rejected_ids.shape)

        # batch size x max sequence x vocab size
        opt_w_logits = self.opt(chosen_ids).logits
        opt_l_logits = self.opt(rejected_ids).logits

        # batch size x max sequence x vocab size
        opt_w_probs = torch.log(F.softmax(opt_w_logits)) * inputs['c_mask'].unsqueeze(-1)
        opt_l_probs = torch.log(F.softmax(opt_l_logits)) * inputs['r_mask'].unsqueeze(-1)

        with torch.no_grad():
            ref_w_logits = self.ref(chosen_ids).logits
            ref_l_logits = self.ref(rejected_ids).logits
            ref_w_probs = torch.log(F.softmax(ref_w_logits)) * inputs['c_mask'].unsqueeze(-1)
            ref_l_probs = torch.log(F.softmax(ref_l_logits)) * inputs['r_mask'].unsqueeze(-1)

        chosen_ids = chosen_ids.unsqueeze(-1)
        rejected_ids = rejected_ids.unsqueeze(-1)

        # print('#### SHAPES HERE ####')
        # print(ref_probs.shape)
        # print(opt_probs.shape)
        # print(rejected_ids.shape)
        # print(chosen_ids.shape)

        pi_ref_w = torch.sum(torch.gather(ref_w_probs, 2, chosen_ids))
        pi_ref_l = torch.sum(torch.gather(ref_l_probs, 2, rejected_ids))
        pi_opt_w = torch.sum(torch.gather(opt_w_probs, 2, chosen_ids))
        pi_opt_l = torch.sum(torch.gather(opt_l_probs, 2, rejected_ids))

        loss = -torch.log(torch.sigmoid(beta * (pi_opt_w - pi_ref_w - pi_opt_l + pi_ref_l)))
        return loss


    def training_step(self, inputs, batch_idx):
        return self.shared_forward(inputs, batch_idx, 'train')

    def validation_step(self, inputs, batch_idx):
        return self.shared_forward(inputs, batch_idx, 'valid')

    def test_step(self, inputs, batch_idx):
        return self.shared_forward(inputs, batch_idx, 'test')


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer