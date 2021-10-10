import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import ElectraModel

from .data import (
    Inputs,
    DiscOutput,
    FakedData,
    features_to_inputs,
    MLMOutput,
    get_updated_inputs,
)
from .helpers import mask, unmask, sample_from_softmax


class PretrainingModel(nn.Module):
    def __init__(self, config):
        super(PretrainingModel, self).__init__()
        self.config = config
        self.generator = ElectraModel.from_pretrained(config.generator_path)
        self.discriminator = ElectraModel.from_pretrained(config.discriminator_path)
        self._tie_weight()
        self.d_linear = nn.Sequential(
            nn.Linear(
                self.discriminator.config.hidden_size,
                self.discriminator.config.hidden_size,
            ),
            nn.GELU(),
            nn.Linear(self.discriminator.config.hidden_size, 1),
        )
        self.g_linear = nn.Sequential(
            nn.Linear(
                self.generator.config.hidden_size, self.generator.config.embedding_size
            ),
            nn.GELU(),
            nn.LayerNorm(self.generator.config.embedding_size),
        )
        self.g_bias = nn.Parameter(
            torch.zeros(self.generator.config.vocab_size), requires_grad=True
        )

    def _expand_embedding(self, size):
        old_embedding = self.discriminator.get_input_embeddings()
        old_embedding_weight = old_embedding.weight.data
        n_embeddings, embedding_dim = old_embedding_weight.shape
        if size <= n_embeddings:
            print(f"Old n_embedding {n_embeddings} not less then {size}.")
            return
        new_embedding = nn.Embedding(
            num_embeddings=size,
            embedding_dim=embedding_dim,
            padding_idx=old_embedding.padding_idx,
        )
        new_embedding.weight.data[:n_embeddings] = old_embedding_weight
        self.discriminator.set_input_embeddings(new_embedding)
        self.generator.config.vocab_size = self.discriminator.config.vocab_size = size
        self._tie_weight()

    def _tie_weight(self):
        self.generator.set_input_embeddings(self.discriminator.get_input_embeddings())

    def _get_masked_inputs(self, features):
        masked_inputs = mask(
            features_to_inputs(features),
            self.config.mask_prob,
            self.config.max_predictions_per_seq,
            self.config.mask_token_id,
            self.config.no_mask_token_ids,
        )
        return masked_inputs

    def forward(self, features, evaluate=False):
        masked_inputs = self._get_masked_inputs(features)

        mlm_output = self._get_masked_lm_output(masked_inputs)
        fake_data = self._get_fake_data(masked_inputs, mlm_output.logits)
        total_loss = self.config.gen_weight * mlm_output.loss

        disc_output = self._get_discriminator_output(
            fake_data.inputs, fake_data.is_fake_tokens
        )
        total_loss += self.config.disc_weight * disc_output.loss

        if evaluate:
            eval_fn_inputs = {
                "input_ids": masked_inputs.input_ids,
                "masked_lm_preds": mlm_output.preds,
                "mlm_loss": mlm_output.loss,
                "masked_lm_ids": masked_inputs.masked_lm_ids,
                "masked_lm_positions": masked_inputs.masked_lm_positions,
                "input_mask": masked_inputs.input_mask.bool(),
                "disc_loss": disc_output.per_example_loss,
                "disc_labels": disc_output.labels.bool(),
                "disc_probs": disc_output.probs,
                "disc_preds": disc_output.preds,
                "sampled_token_ids": fake_data.sampled_token_ids,
            }
            return (
                total_loss,
                eval_fn_inputs,
            )
        return (total_loss,)

    def eval_metrics(self, **kwargs):
        """Computes the loss and accuracy of the model."""
        d = kwargs
        metrics = dict()
        metrics["masked_lm_accuracy"] = (
            (d["masked_lm_ids"] == d["masked_lm_preds"]).float().mean()
        ).item()
        metrics["masked_lm_loss"] = d["mlm_loss"].item()
        metrics["sampled_masked_lm_accuracy"] = (
            (d["masked_lm_ids"] == d["sampled_token_ids"]).float().mean()
        ).item()
        metrics["disc_loss"] = d["disc_loss"].mean().item()
        metrics["disc_auc"] = roc_auc_score(
            d["disc_labels"][d["input_mask"]].detach().cpu(),
            d["disc_probs"][d["input_mask"]].detach().cpu(),
        )
        metrics["disc_accuracy"] = (
            (d["disc_labels"] == d["disc_preds"])[d["input_mask"]].float().mean()
        ).item()
        metrics["disc_precision"] = (
            (d["disc_labels"] == d["disc_preds"])[d["input_mask"] & d["disc_preds"]]
            .float()
            .mean()
        ).item()
        metrics["disc_recall"] = (
            (d["disc_labels"] == d["disc_preds"])[d["input_mask"] & d["disc_labels"]]
            .float()
            .mean()
        ).item()
        metrics["disc_f1"] = (
            2
            * metrics["disc_precision"]
            * metrics["disc_recall"]
            / max(metrics["disc_recall"] + metrics["disc_precision"], 1e-12)
        )

        """ Additional Data """
        metrics["TP"] = (
            (d["disc_labels"] * d["disc_preds"])[d["input_mask"]].sum().item()
        )
        metrics["TN"] = (
            ((~d["disc_labels"]) * (~d["disc_preds"]))[d["input_mask"]].sum().item()
        )
        metrics["FP"] = (
            ((~d["disc_labels"]) * d["disc_preds"])[d["input_mask"]].sum().item()
        )
        metrics["FN"] = (
            (d["disc_labels"] * (~d["disc_preds"]))[d["input_mask"]].sum().item()
        )

        return metrics

    def _get_masked_lm_output(self, inputs: Inputs):
        relevant_hidden = self.generator(
            input_ids=inputs.input_ids, attention_mask=inputs.input_mask
        ).last_hidden_state[
            inputs.masked_lm_positions
        ]  # K(n_masked_in_batch) x D_hidden
        hidden = self.g_linear(relevant_hidden)  # K x D_emb
        logits = (
            hidden @ self.generator.embeddings.word_embeddings.weight.T
        ) + self.g_bias  # K x |V|

        log_probs = torch.log_softmax(logits, dim=-1)  # K x |V|
        label_log_probs = F.nll_loss(input=log_probs, target=inputs.masked_lm_ids)  # K

        return MLMOutput(
            logits=logits,
            probs=torch.softmax(logits, dim=-1),
            per_example_loss=label_log_probs,
            loss=label_log_probs.mean(),
            preds=torch.argmax(log_probs, dim=-1).long(),
        )

    def _get_discriminator_output(self, inputs, labels):
        hidden = self.discriminator(
            input_ids=inputs.input_ids, attention_mask=inputs.input_mask
        ).last_hidden_state  # B x L x D_hidden
        logits = self.d_linear(hidden).squeeze(-1)  # B x L
        masked = inputs.input_mask.float()  # B x L
        losses = F.binary_cross_entropy_with_logits(
            logits, labels.float(), weight=masked, reduction="none"
        )  # B x L

        return DiscOutput(
            loss=losses.sum() / masked.sum().clip(1),
            per_example_loss=losses.sum(-1) / masked.sum(-1).clip(1),
            probs=torch.sigmoid(logits),
            preds=(logits > 0),
            labels=labels,
        )

    def _get_fake_data(self, inputs, mlm_logits):
        inputs = unmask(inputs)

        sampled_token_ids = sample_from_softmax(
            mlm_logits / self.config.temperature
        ).detach()  # tokens to be replaced with
        updated_input_ids = inputs.input_ids.clone()
        updated_input_ids[
            inputs.masked_lm_positions
        ] = sampled_token_ids  # replace with fake-sampled token

        labels = (updated_input_ids != inputs.input_ids).float()
        updated_inputs = get_updated_inputs(inputs, input_ids=updated_input_ids)
        return FakedData(
            inputs=updated_inputs,
            is_fake_tokens=labels,
            sampled_token_ids=sampled_token_ids,
        )
