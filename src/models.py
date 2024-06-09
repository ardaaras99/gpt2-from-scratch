import torch
import torch.nn as nn

from src import GPT2Config
from src.backbones.gpt import GPT
from src.heads.classification import ClassificationHead


class GPTClassifier(nn.Module):
    def __init__(self, gpt_config: GPT2Config, num_classes: int = None):
        super().__init__()
        self.gpt = GPT(gpt_config)
        if num_classes is None:
            print("num_classes is not provided, assuming pre-trainig mode")
            num_classes = gpt_config.vocab_size
        self.head = ClassificationHead(gpt_config.emb_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.gpt(input_ids)
        x = self.head(x)
        return x
