import torch
import random
from torch.utils.data import Dataset
from transformers import ElectraTokenizer
from glob import glob
from tqdm import tqdm


class KoELECTRADataset(Dataset):
    data = None

    def __init__(self, config, data_path, **kwargs):
        config.update(kwargs)
        self.config = config
        self.data_path = data_path
        self.tokenizer = ElectraTokenizer.from_pretrained(config.tokenizer_path)
        self.parse_files()

    def _parse_one_file(self, file_dir):
        articles = []
        with open(file_dir, "r") as f:
            lines = f.readlines()

        article = []
        for line in lines:
            if line == "\n":
                if article:
                    articles.append(article)
                    article = []
            else:
                if len(line) >= 20:
                    article.append(line.strip())

        articles.append(article)
        return articles

    def parse_files(self):
        file_dirs = sorted(glob(str(self.data_path / "*.txt")))
        articles = []
        for file_dir in tqdm(file_dirs):
            articles.extend(self._parse_one_file(file_dir))
        self.data = articles
        return articles

    def indexify(self, article):
        token_ids = [self.tokenizer.cls_token_id]
        for sentence in article:
            token_ids.extend(
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence))
                + [self.tokenizer.sep_token_id]
            )
        return token_ids[: self.config.max_seq_length]

    def sample_subtext(self, article, num_samples):
        length = len(article)
        start_index = random.randint(0, length - num_samples)
        return article[start_index : start_index + num_samples]

    def __getitem__(self, item):
        article = self.data[item]
        if len(article) < 8:
            token_ids = self.indexify(article)
        # Sample (adjacent) 8~12 sentences **including title**
        else:
            token_ids = self.indexify(
                [article[0]]
                + self.sample_subtext(article[1:], random.randint(7, len(article) - 1))
            )
        return torch.LongTensor(token_ids)

    def __len__(self):
        return len(self.data)
