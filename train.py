from pathlib import Path

import torch
import wandb
from dotmap import DotMap
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_pretrain.dataloader import get_collate_fn
from torch_pretrain.dataset import KoELECTRADataset
from torch_pretrain.model import PretrainingModel
from torch_pretrain.optimizer import get_optimizer
from torch_pretrain.utils import WeightAverager, MultiAverager

config = DotMap(
    max_seq_length=512,
    tokenizer_path="monologg/koelectra-small-v3-discriminator",
    generator_path="monologg/koelectra-small-v3-generator",
    discriminator_path="monologg/koelectra-small-v3-discriminator",
    mask_prob=0.15,
    max_predictions_per_seq=int((0.15 + 0.005) * 512),
    mask_token_id=4,
    pad_token_id=0,
    no_mask_token_ids=[0, 1, 2, 3, 4],  # ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
    gen_weight=1.0,
    disc_weight=50.0,
    temperature=1.0,
    train_batch_size=48,
    eval_batch_size=48,
    learning_rate=5e-4,
    lr_decay_power=1.0,
    warmup_proportion=0.1,
    num_train_steps=100000,
    num_eval_steps=10,
    logging_interval=100,
    saving_interval=1000,
    cuda=True,
)

dataset = KoELECTRADataset(config, Path("./data/"))
dataloader = DataLoader(
    dataset,
    batch_size=config.train_batch_size,
    shuffle=True,
    drop_last=False,
    collate_fn=get_collate_fn(config),
)
val_dataloader = DataLoader(
    dataset,
    batch_size=config.eval_batch_size,
    shuffle=True,
    drop_last=False,
    collate_fn=get_collate_fn(config),
)

model = PretrainingModel(config).cuda()

optimizer, scheduler = get_optimizer(model, config, config.num_train_steps)

model.train()

runpath = Path("./experiment") / "train"
runpath.mkdir(parents=True, exist_ok=True)

wandb.init(project="KoElectraFinetune", config=config)
wandb.run.name = "train"
train_step = 0
for ep in range(config.num_train_steps // len(dataloader) + 1):
    tdl = tqdm(dataloader)
    for i, batch in enumerate(tdl):
        optimizer.zero_grad()
        (loss,) = model(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        tdl.set_description(
            f"EP {ep:2d}| Loss {loss.item():.4f}",
            refresh=False,
        )
        train_step += 1
        if (train_step - 1) * (train_step % config.logging_interval) == 0:
            wandb.log({"step": train_step, "train_loss": loss.item()})

        if (train_step - 1) * (
            train_step % config.saving_interval
        ) == 0 or train_step == config.num_train_steps:
            model.eval()
            eval_step = 0
            with torch.no_grad():
                logging_summary = MultiAverager(WeightAverager)
                tvdl = tqdm(val_dataloader)
                for _, val_batch in enumerate(tvdl):
                    loss, eval_inputs = model(val_batch, evaluate=True)
                    batch_summary = model.eval_metrics(**eval_inputs)
                    batch_summary.update({"val_loss": loss.item()})
                    logging_summary.add(batch_summary, len(val_batch["input_ids"]))
                    tvdl.set_description(
                        "\t".join(
                            f"{k}: {getattr(logging_summary, k).get():.4f}"
                            for k in logging_summary
                        ),
                        refresh=False,
                    )
                    eval_step += 1
                    if eval_step == config.num_eval_steps:
                        break
                logging_summary = logging_summary.get()
                logging_summary.update({"step": train_step})
                wandb.log(logging_summary)
            svdir = runpath / str(train_step)
            svdir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), svdir / f"model.pth")
            torch.save(optimizer.state_dict(), svdir / f"optim.pth")
            torch.save(scheduler.state_dict(), svdir / f"scheduler.pth")
            model.train()

            if train_step == config.num_train_steps:
                break

print("Bye~")
