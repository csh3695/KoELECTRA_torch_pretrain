import json
from pathlib import Path

import torch
import wandb
from dotmap import DotMap
from torch.utils.data import DataLoader
from tqdm import tqdm

from hate_speech.dataset import KoreanHateSpeechDataset
from hate_speech.model import ElectraForBiasClassification
from hate_speech.optimizer import get_optimizer
from hate_speech.utils import MultiAverager, WeightAverager

run_name = "hatespeech-1"


config = DotMap(
    tokenizer_path="monologg/koelectra-small-v3-discriminator",
    electra_path="monologg/koelectra-small-v3-discriminator",
    max_length=128,
    train_batch_size=256,
    val_batch_size=256,
    bias_loss_coef=0.5,
    hate_loss_coef=1.0,
    gender_loss_coef=0.5,
    curse_loss_coef=2.0,
    dropout=0.1,
    learning_rate=5e-5,
    warmup_proportion=0.1,
)


train_dataset = KoreanHateSpeechDataset(config, "train")
curse_dataset = KoreanHateSpeechDataset(config, "curse")

val_dataset = KoreanHateSpeechDataset(config, "dev")
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.train_batch_size,
    collate_fn=train_dataset.collate_fn,
    shuffle=True,
)
curse_dataloader = DataLoader(
    curse_dataset,
    batch_size=config.train_batch_size,
    collate_fn=curse_dataset.collate_fn,
    shuffle=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config.val_batch_size,
    collate_fn=val_dataset.collate_fn,
    shuffle=False,
)


model = ElectraForBiasClassification(
    config, train_dataset.bias_label_map, train_dataset.hate_label_map
)

if config.cuda:
    model = model.cuda()

optimizer, scheduler = get_optimizer(
    model, config, 10 * (len(train_dataloader) + len(curse_dataloader))
)

model.train()

runpath = Path("./experiment") / run_name
runpath.mkdir(parents=True, exist_ok=True)
with open(runpath / "config.json", "w") as f:
    json.dump(config.toDict(), f)

wandb.init(project="KoElectraHateSpeech", config=config)
wandb.run.name = run_name

for ep in range(10):
    tdl = tqdm(curse_dataloader)
    curse_loss = WeightAverager()
    for i, batch in enumerate(tdl):
        optimizer.zero_grad()
        loss = model(**batch)[0]
        curse_loss.add(loss.item(), len(batch["input_ids"]))
        loss.backward()
        optimizer.step()
        scheduler.step()
        tdl.set_description(
            f"EP {ep:2d}| Curse Loss {loss.item():.4f}",
            refresh=False,
        )
    wandb.log({"ep": ep, "train_curse_loss": curse_loss.get()})

    tdl = tqdm(train_dataloader)
    hate_loss = WeightAverager()
    for i, batch in enumerate(tdl):
        optimizer.zero_grad()
        loss = model(**batch)[0]
        hate_loss.add(loss.item(), len(batch["input_ids"]))
        loss.backward()
        optimizer.step()
        scheduler.step()
        tdl.set_description(
            f"EP {ep:2d}| Hate Loss {loss.item():.4f}",
            refresh=False,
        )
    wandb.log({"ep": ep, "train_hate_loss": hate_loss.get()})

    tdl = tqdm(val_dataloader)
    logging_summary = MultiAverager(WeightAverager)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tdl):
            result = model(batch, evaluate=True)
            loss = result[0]
            batch_summary = result[-1]
            batch_summary.update({"val_hate_loss": loss.item()})
            logging_summary.add(batch_summary, len(batch["input_ids"]))
            tdl.set_description(
                "\t".join(
                    f"{k}: {getattr(logging_summary, k).get():.4f}"
                    for k in logging_summary
                ),
                refresh=False,
            )

    logging_summary = logging_summary.get()
    logging_summary.update({"ep": ep})
    wandb.log(logging_summary)

    svdir = runpath / str(ep)
    svdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), svdir / f"model.pth")
    torch.save(optimizer.state_dict(), svdir / f"optim.pth")
    torch.save(scheduler.state_dict(), svdir / f"scheduler.pth")
    model.train()

print("Bye~")
