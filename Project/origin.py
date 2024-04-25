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

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"],batch['flipped_input_ids'] = self.torch_mask_tokens(
                batch["input_ids"], batch['flipped_input_ids'],special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, flipped_inputs,special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
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
        flipped_inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        flipped_inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels,flipped_inputs

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        flipped_inputs = inputs.pop("flipped_input_ids")
        lambda_ = 0.5
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.prediction_logits
        loss_fct = torch.nn.CrossEntropyLoss()
        mse = torch.nn.MSELoss()
        loss = loss_fct(logits.transpose(1,2),labels)
        flipped_outputs = model(input_ids=flipped_inputs)
        flipped_logits = flipped_outputs.prediction_logits
        flipped_loss = mse(logits,flipped_logits)
        logs = {"loss": loss.detach().cpu().item(), "flipped_loss": flipped_loss.detach().cpu().item()}
        self.log(logs)
        loss = loss+lambda_*flipped_loss
        return (loss, outputs) if return_outputs else loss