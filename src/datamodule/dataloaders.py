from torch.utils.data import DataLoader

from src.datamodule.datasets import GPTPreTrainDataset, PreTrainDatasetConfig


def get_dataloader(dataset_cfg: PreTrainDatasetConfig, batch_size: int = 4):
    dataset = GPTPreTrainDataset(dataset_cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return dataloader, dataset.tokenizer
