from torch.nn.utils.rnn import pad_sequence


def get_collate_fn(config):
    def collate_fn(samples):
        batch = dict()
        batch["input_ids"] = pad_sequence(
            samples, batch_first=True, padding_value=config.pad_token_id
        )
        batch["input_mask"] = (batch["input_ids"] != config.pad_token_id).float()
        if config.cuda:
            batch = dict((k, batch[k].cuda()) for k in batch)
        return batch

    return collate_fn
