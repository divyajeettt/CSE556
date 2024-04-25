# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%
import torch
from transformers import BertForPreTraining, BertTokenizerFast, Trainer, TrainingArguments,DataCollatorForLanguageModeling
import numpy as np
from typing import Union, List, Dict,Mapping, Optional, Tuple,Any
import pandas as pd
from datasets import Dataset
from flipper import Flipper

# %%
dataset = pd.read_csv('corpus.csv')
dataset = dataset.dropna()
dataset = Dataset.from_pandas(dataset)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
flipper = Flipper('gendered_words/gendered_words.json')
flipper.process_tokenizer(tokenizer)

def tokenize(batch):
    inputs = tokenizer(batch['original'], truncation=True, padding='max_length',max_length=128)
    #inputs['labels'] = flipper.flip_label(batch['original'])
    return inputs

dataset = dataset.map(tokenize, batched=False)

# %%
class TextData(torch.utils.data.Dataset):
    def __init__(self):
        pass

# %%
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        flipped_inputs = inputs.pop("flipped_input_ids")
        lambda_ = 1
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.prediction_logits
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.transpose(1,2),labels)
        loss_flipped = loss_fct(logits.transpose(1,2),flipped_inputs)
        logs = {"loss": loss.detach().cpu().item(), "flipped_loss": loss_flipped.detach().cpu().item()}
        self.log(logs)
        loss = loss+lambda_*loss_flipped
        return (loss, outputs) if return_outputs else loss

# %%
def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


class CustomCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of),
            }
        flipped_text = [flipper.flip(self.tokenizer.decode(text,skip_special_tokens=True)) for text in batch['input_ids']]
        batch['flipped_input_ids'] = [self.tokenizer.encode_plus(t, padding="max_length", truncation=True, max_length=128,return_tensors='pt')["input_ids"] for t in flipped_text]
        batch['flipped_input_ids'] = torch.cat(batch['flipped_input_ids'])
        sentences = [self.tokenizer.decode(text,skip_special_tokens=True) for text in batch['input_ids']]
        labels = flipper.process_tensor(batch['input_ids'])
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if torch.rand(1).item() < 0.5:
            labels = None
        if self.mlm:
            batch["input_ids"], batch["labels"], batch['flipped_input_ids'] = self.torch_mask_tokens(
                batch["input_ids"], batch['flipped_input_ids'],special_tokens_mask=special_tokens_mask,
                def_ind = labels
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, flipped_inputs,special_tokens_mask: Optional[Any] = None,def_ind=None) -> Tuple[Any, Any]:
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
        masked_indices = def_ind if def_ind is not None else torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        flipped_inputs[~masked_indices] = -100  # We only compute loss on masked tokens

        if def_ind is not None:
            inputs[def_ind] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            return inputs, labels,flipped_inputs

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        flipped_inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        flipped_inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels,labels

# %%
model = BertForPreTraining.from_pretrained('bert-base-uncased')
training_args = TrainingArguments(
    output_dir='./results3',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    logging_dir='./logs',            # directory for storing logs
    fp16=True,
    learning_rate=5e-5,
    report_to='none'
)
datasets = dataset.train_test_split(test_size=0.1)
trainer = CustomTrainer(
    model=model,
    args=training_args,
    data_collator=CustomCollator(tokenizer,mlm=True, mlm_probability=0.15),
    train_dataset=datasets['train'],
    eval_dataset=datasets['test']
)

# %%
trainer.save_model('model')
trainer.train()        
trainer.save_model('model')


