from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.tokenizers.models import BPE


@dataclass
class RawTextDatasetConfig:
    file_path: str
    train_ratio: float = 0.90
    val_ratio: float = 0.10


class RawTextDataset:
    def __init__(self, cfg: RawTextDatasetConfig):
        self.cfg = cfg
        self.text_data = self.read_file()
        self.train_data, self.val_data = self.split_data()

    def read_file(self):
        with open(self.cfg.file_path, encoding="utf-8") as file:
            text_data = file.read()
        return text_data

    def split_data(self):
        split_idx = int(len(self.text_data) * self.cfg.train_ratio)
        train_data = self.text_data[:split_idx]
        val_data = self.text_data[split_idx:]
        return train_data, val_data


@dataclass
class PreTrainDatasetConfig:
    tokenizer_name: str
    max_length: int
    stride: int
    mode: str
    raw_dataset_cfg: RawTextDatasetConfig


class GPTPreTrainDataset(Dataset):
    def __init__(self, cfg: PreTrainDatasetConfig):
        self.cfg = cfg
        self.tokenizer = BPE(name=cfg.tokenizer_name)
        txt = self.select_txt()
        self.input_ids, self.target_ids = self.get_token_ids(txt)

    def get_token_ids(self, txt: str):
        input_ids, target_ids = [], []
        token_ids = self.tokenizer.encode(txt)
        for i in range(0, len(token_ids) - self.cfg.max_length, self.cfg.stride):
            input_chunk = token_ids[i : i + self.cfg.max_length]
            target_chunk = token_ids[i + 1 : i + self.cfg.max_length + 1]
            input_ids.append(torch.tensor(input_chunk))
            target_ids.append(torch.tensor(target_chunk))
        return input_ids, target_ids

    def select_txt(self):
        if self.cfg.mode == "train":
            txt = RawTextDataset(self.cfg.raw_dataset_cfg).train_data
        elif self.cfg.mode == "val":
            txt = RawTextDataset(self.cfg.raw_dataset_cfg).val_data
        else:
            raise ValueError("mode should be either 'train' or 'val'")

        return txt

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
