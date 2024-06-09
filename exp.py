# %%
import torch

from engine.engine import Engine, calc_loss_loader
from src import GPT_CONFIG_124M
from src.datamodule.dataloaders import get_dataloader
from src.datamodule.datasets import PreTrainDatasetConfig, RawTextDatasetConfig
from src.models import GPTClassifier
from src.utils.model_summary import plot_losses

torch.manual_seed(123)
model = GPTClassifier(GPT_CONFIG_124M)
model.eval()

model_cfg = GPT_CONFIG_124M

raw_dataset_cfg = RawTextDatasetConfig(
    file_path="raw-datas/the-verdict.txt", train_ratio=0.9, val_ratio=0.1
)

dataset_cfg = PreTrainDatasetConfig(
    tokenizer_name="gpt2",
    max_length=model_cfg.context_length,
    stride=model_cfg.context_length,
    mode="train",
    raw_dataset_cfg=raw_dataset_cfg,
)

train_loader, _ = get_dataloader(dataset_cfg, batch_size=2)

dataset_cfg = PreTrainDatasetConfig(
    tokenizer_name="gpt2",
    max_length=model_cfg.context_length,
    stride=model_cfg.context_length,
    mode="val",
    raw_dataset_cfg=raw_dataset_cfg,
)
val_loader, _ = get_dataloader(dataset_cfg, batch_size=2)


device = "mps"
model.to(device)

with torch.no_grad():  # B
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Initial Training loss:", train_loss)
print("Initial Validation loss:", val_loss)

engine = Engine(
    model=model,
    loaders={"train": train_loader, "val": val_loader},
    optimizer=torch.optim.AdamW(model.parameters(), lr=0.0004),
    device="mps",
    # Parameters for the TokenDecoder
    max_new_tokens=25,
    temperature=0.0,
    top_k=None,
    start_context="Every effort moves you",
)
num_epochs = 10
engine.pipeline(num_epochs=num_epochs)

train_losses = engine.train_losses
val_losses = engine.val_losses
tokens_seen = engine.track_tokens_seen

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# %%
# import tiktoken
# from src.utils.text_generation import (
#     greedy_decode,
#     text_to_token_ids,
#     token_ids_to_text,
# )

# tokenizer = tiktoken.get_encoding("gpt2")
# token_ids = greedy_decode(
#     model=model,
#     input_ids=text_to_token_ids("Every effort moves you", tokenizer).to(device),
#     max_new_tokens=25,
#     context_length=GPT_CONFIG_124M.context_length,
# )
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# %%


# %%
