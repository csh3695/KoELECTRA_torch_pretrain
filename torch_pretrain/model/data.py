import collections

Inputs = collections.namedtuple(
    "Inputs",
    [
        "input_ids",
        "input_mask",
        "segment_ids",
        "masked_lm_positions",
        "masked_lm_ids",
    ],
)
MLMOutput = collections.namedtuple(
    "MLMOutput", ["logits", "probs", "loss", "per_example_loss", "preds"]
)
DiscOutput = collections.namedtuple(
    "DiscOutput", ["loss", "per_example_loss", "probs", "preds", "labels"]
)
FakedData = collections.namedtuple(
    "FakedData", ["inputs", "is_fake_tokens", "sampled_token_ids"]
)


def features_to_inputs(features):
    return Inputs(
        input_ids=features["input_ids"],
        input_mask=features["input_mask"],
        segment_ids=features["segment_ids"] if "segment_ids" in features else None,
        masked_lm_positions=(
            features["masked_lm_positions"]
            if "masked_lm_positions" in features
            else None
        ),
        masked_lm_ids=(
            features["masked_lm_ids"] if "masked_lm_ids" in features else None
        ),
    )


def get_updated_inputs(inputs, **kwargs):
    features = inputs._asdict()
    for k, v in kwargs.items():
        features[k] = v
    return features_to_inputs(features)
