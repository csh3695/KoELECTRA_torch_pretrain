from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def get_optimizer(model, config, total_steps):
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.0,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * config.warmup_proportion),
        num_training_steps=total_steps,
    )
    return optimizer, scheduler
