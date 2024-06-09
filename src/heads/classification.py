import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, emb_dim, num_classes, bias=False):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)
