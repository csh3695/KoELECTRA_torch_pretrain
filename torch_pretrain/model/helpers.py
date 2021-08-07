from .data import get_updated_inputs
import torch


def mask(
    inputs,
    mask_prob,
    max_predictions_per_seq,
    mask_token_id,
    no_mask_token_ids,
):
    N = max_predictions_per_seq
    candidate_to_mask = inputs.input_mask.clone().bool()
    for excluded_token_id in no_mask_token_ids:
        candidate_to_mask &= inputs.input_ids != excluded_token_id

    candidate_sampling_val = (
        torch.zeros_like(candidate_to_mask).float().uniform_(1, 2) * candidate_to_mask
    )

    num_tokens = inputs.input_mask.sum(-1)  # B x L -> B
    num_to_predict = torch.round(num_tokens * mask_prob).clip(1, N)  # B

    masked_lm_positions = candidate_sampling_val.argsort(
        dim=-1, descending=True
    ).argsort(dim=-1) < num_to_predict.unsqueeze(-1)
    masked_lm_ids = inputs.input_ids[masked_lm_positions].clone()

    replace_with_mask_positions = masked_lm_positions * (
        torch.zeros_like(masked_lm_positions).float().uniform_() < 0.85
    )
    inputs_ids = inputs.input_ids.clone().masked_fill_(
        replace_with_mask_positions, mask_token_id
    )

    return get_updated_inputs(
        inputs,
        input_ids=inputs_ids.detach(),
        masked_lm_positions=masked_lm_positions,
        masked_lm_ids=masked_lm_ids,
    )


def unmask(inputs):
    unmasked_input_ids = inputs.input_ids.clone()
    unmasked_input_ids[inputs.masked_lm_positions] = inputs.masked_lm_ids.clone()
    return get_updated_inputs(inputs, input_ids=unmasked_input_ids)


def sample_from_softmax(logits):
    uniform_noise = torch.zeros_like(logits).uniform_()
    gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-9) + 1e-9)
    token_ids = torch.argmax(torch.softmax(logits + gumbel_noise, dim=-1), dim=-1)
    return token_ids
