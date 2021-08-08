import argparse
from time import strftime
from dotmap import DotMap


class PretrainArgParser:
    config_keys = [
        "max_seq_length",
        "tokenizer_path",
        "generator_path",
        "discriminator_path",
        "mask_prob",
        "max_predictions_per_seq",
        "mask_token_id",
        "pad_token_id",
        "no_mask_token_ids",
        "gen_weight",
        "disc_weight",
        "temperature",
        "train_batch_size",
        "eval_batch_size",
        "learning_rate",
        "lr_decay_power",
        "warmup_proportion",
        "num_train_steps",
        "num_eval_steps",
        "logging_interval",
        "saving_interval",
        "cuda",
    ]

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Arguments Parser for Pre-training KoELECTRA."
        )
        self.parser.add_argument(
            "--run_name", type=str, default=strftime("%y%m%d_%H%M%S")
        )
        self.parser.add_argument("--cuda_id", type=int, default=0)

        self.parser.add_argument("--max_seq_length", type=int, default=512)
        self.parser.add_argument(
            "--tokenizer_path",
            type=str,
            default="monologg/koelectra-small-v3-discriminator",
        )
        self.parser.add_argument(
            "--generator_path",
            type=str,
            default="monologg/koelectra-small-v3-generator",
        )
        self.parser.add_argument(
            "--discriminator_path",
            type=str,
            default="monologg/koelectra-small-v3-discriminator",
        )
        self.parser.add_argument("--mask_prob", type=float, default=0.15)
        self.parser.add_argument(
            "--max_predictions_per_seq", type=int, default=int((0.15 + 0.005) * 512)
        )
        self.parser.add_argument("--mask_token_id", type=int, default=4)
        self.parser.add_argument("--pad_token_id", type=int, default=0)
        self.parser.add_argument(
            "--no_mask_token_ids", type=int, nargs="+", default=[0, 1, 2, 3, 4]
        )  # ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
        self.parser.add_argument("--gen_weight", type=float, default=1.0)
        self.parser.add_argument("--disc_weight", type=float, default=50.0)
        self.parser.add_argument("--temperature", type=float, default=1.0)
        self.parser.add_argument("--train_batch_size", type=int, default=48)
        self.parser.add_argument("--eval_batch_size", type=int, default=48)
        self.parser.add_argument("--learning_rate", type=float, default=0.0005)
        self.parser.add_argument("--lr_decay_power", type=float, default=1.0)
        self.parser.add_argument("--warmup_proportion", type=float, default=0.1)
        self.parser.add_argument("--num_train_steps", type=int, default=100000)
        self.parser.add_argument("--num_eval_steps", type=int, default=10)
        self.parser.add_argument("--logging_interval", type=int, default=100)
        self.parser.add_argument("--saving_interval", type=int, default=1000)
        self.parser.add_argument("--cuda", type=bool, default=True)

    def parse_args(self):
        args = self.parser.parse_args()
        argdict = vars(args)
        return args, DotMap(dict((k, argdict[k]) for k in self.config_keys))
