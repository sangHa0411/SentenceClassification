
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
from transformers.file_utils import PaddingStrategy
from transformers import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

@dataclass
class DataCollatorForMaskPadding(DataCollatorForLanguageModeling):
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def torch_call(self, examples) :
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(examples, 
            padding=True, 
            return_tensors="pt", 
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        if self.mlm:
            batch["input_ids"], batch["mask_labels"] = self.torch_mask_tokens(inputs=batch["input_ids"], 
                special_tokens_mask=special_tokens_mask
            )
        return batch

    def torch_mask_tokens(self, inputs, special_tokens_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

@dataclass
class DataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        decoder_input_ids = [feature["decoder_input_ids"] for feature in features] if "decoder_input_ids" in features[0].keys() else None

        pad_to_multiple_of = 1 if self.pad_to_multiple_of is None else self.pad_to_multiple_of
        if decoder_input_ids is not None:
            max_decoder_input_length = max(len(l) for l in decoder_input_ids)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                if max_decoder_input_length % pad_to_multiple_of != 0 :
                    max_decoder_input_length = ((max_decoder_input_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

                remainder = [self.tokenizer.pad_token_id] * (max_decoder_input_length - len(feature["decoder_input_ids"]))
                if isinstance(feature["decoder_input_ids"], list):
                    feature["decoder_input_ids"] = (
                        feature["decoder_input_ids"] + remainder if padding_side == "right" else remainder + feature["decoder_input_ids"]
                    )
                elif padding_side == "right":
                    feature["decoder_input_ids"] = np.concatenate([feature["decoder_input_ids"], remainder]).astype(np.int64)
                else:
                    feature["decoder_input_ids"] = np.concatenate([remainder, feature["decoder_input_ids"]]).astype(np.int64)
        
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch