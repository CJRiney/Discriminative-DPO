from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers import DataCollatorForLanguageModeling
from dataclasses import dataclass
import numpy as np
import torch
import warnings

from typing import Union, Optional, List, Dict, Any

@dataclass
class DataCollatorWithPaddingSFT:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    inst_prefix = ""
    response_token_ids: List[int] = None
    ignore_index = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        device='cuda'

        batch = {}
        input_count = len(features)
        length_chosens = [len(x['chosen_ids']) for x in features]
        length_rejects = [len(x['rejected_ids']) for x in features]
        c_max_length = max(length_chosens)
        r_max_length = max(length_rejects)

        chosen_ids = []
        rejected_ids = []
        for i in range(input_count):
            c_curr_ids = features[i]['chosen_ids']
            r_curr_ids = features[i]['rejected_ids']

            c_pad_input_ids = c_curr_ids + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id] * (c_max_length - len(c_curr_ids))
            r_pad_input_ids = r_curr_ids + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id] * (r_max_length - len(r_curr_ids))

            chosen_ids.append(c_pad_input_ids)
            rejected_ids.append(r_pad_input_ids)

        chosen_ids = torch.LongTensor(chosen_ids)
        rejected_ids = torch.LongTensor(rejected_ids)

        c_mask_list = []
        r_mask_list = []
        for i in range(input_count):
            c_mask_curr = (chosen_ids[i] != self.tokenizer.pad_token_id).float()
            r_mask_curr = (rejected_ids[i] != self.tokenizer.pad_token_id).float()

            c_mask_curr[:features[i]['c_start_idx']] = 0
            r_mask_curr[:features[i]['r_start_idx']] = 0

            c_mask_list.append(c_mask_curr.unsqueeze(0))
            r_mask_list.append(r_mask_curr.unsqueeze(0))

        c_mask = torch.cat(c_mask_list)
        r_mask = torch.cat(r_mask_list)

        batch = {
            'chosen_ids': chosen_ids,
            'rejected_ids': rejected_ids,
            'c_mask': c_mask,
            'r_mask': r_mask
        }

        return batch

