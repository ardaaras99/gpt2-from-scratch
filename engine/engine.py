import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models import GPTClassifier
from src.utils.token_generator import TokenDecoder, text_to_token_ids, token_ids_to_text


def calc_loss_batch(
    input_batch: torch.Tensor, target_batch: torch.Tensor, model: nn.Module, device: str
):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    num_batches = get_num_batches(data_loader, num_batches)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def get_num_batches(data_loader, num_batches=None):

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    return num_batches


class Engine:
    def __init__(
        self,
        model: GPTClassifier,
        loaders: dict[str, DataLoader],
        optimizer: torch.optim.Optimizer,
        device: str,
    ):
        self.model = model.to(device)
        self.train_loader, self.val_loader = loaders["train"], loaders["val"]
        self.optimizer = optimizer
        self.device = device
        self.train_losses, self.val_losses = [], []

        self.track_tokens_seen = []
        self.tokens_seen = 0

    def pipeline(self, num_epochs):
        t = tqdm(range(num_epochs), leave=False)
        for epoch in t:
            self.train_epoch()
            train_loss, val_loss = self.evaluate()
            self.append_metrics(train_loss, val_loss)

            decoded_text = self.get_decoded_text(
                tiktoken.get_encoding("gpt2"), self.device, "Every effort moves you"
            )

            t.set_description(
                f"Ep {epoch+1}: Train loss {train_loss:.3f}, Val loss {val_loss:.3f} Generated Text: {decoded_text}"
            )

    def train_epoch(self):
        # tokens_seen = 0
        # global_step = -1
        self.model.train()
        for input_batch, target_batch in self.train_loader:
            self.optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, self.model, self.device)
            loss.backward()
            self.optimizer.step()
            self.tokens_seen += input_batch.numel()
            # global_step += 1
            # tokens_seen += input_batch.numel()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            train_loss = calc_loss_loader(self.train_loader, self.model, self.device)
            val_loss = calc_loss_loader(self.val_loader, self.model, self.device)
        return train_loss, val_loss

    def append_metrics(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.track_tokens_seen.append(self.tokens_seen)

    def get_decoded_text(self, tokenizer, device, start_context):
        self.model.eval()
        context_length = self.model.gpt.emb_layer.pos_embedding.weight.shape[0]
        input_ids = text_to_token_ids(start_context, tokenizer).to(device)
        dec = TokenDecoder(self.model, input_ids, context_length)
        with torch.no_grad():
            token_ids = dec.generator(max_new_tokens=20, temperature=1.1, top_k=10)
            decoded_text = token_ids_to_text(token_ids, tokenizer)
            return decoded_text.replace("\n", " ")
