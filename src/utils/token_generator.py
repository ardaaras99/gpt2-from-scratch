import torch

from src.tokenizers.models import BPE


class TokenDecoder:
    def __init__(self, model, input_ids, context_length):
        self.model = model
        self.input_ids = input_ids
        self.context_length = context_length
        self.eos_id = 50256

    def greedy(self, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = self.input_ids[:, -self.context_length :]
            with torch.no_grad():
                logits = self.model(idx_cond)
                next_token = torch.argmax(logits[:, -1], dim=-1)
                self.input_ids = torch.cat(
                    [self.input_ids, next_token.unsqueeze(0)], dim=-1
                )

        return self.input_ids

    def generator(self, max_new_tokens, temperature=0.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = self.input_ids[:, -self.context_length :]
            with torch.no_grad():
                logits = self.model(idx_cond)

            logits = logits[:, -1, :]
            logits = self.apply_top_k(logits, top_k)
            logits = self.apply_temperature(logits, temperature)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token == self.eos_id:
                break
            self.input_ids = torch.cat((self.input_ids, next_token), dim=-1)

        return self.input_ids

    @staticmethod
    def apply_top_k(logits, top_k):
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_logit_val = top_logits[:, -1]

            condition = logits < min_logit_val
            mask = torch.tensor(float("-inf")).to(logits.device)
            logits = torch.where(condition, mask, logits)
            return logits
        else:
            return logits

    @staticmethod
    def apply_temperature(logits, temperature):
        if temperature > 0.0:
            logits = logits / temperature
            return logits
        else:
            return logits


def text_to_token_ids(text, tokenizer: BPE) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # .unsqueeze(0) adds the batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer: BPE) -> list[str]:
    flat = token_ids.squeeze(0)  # Remove batch dimension
    return tokenizer.decode(flat.tolist())
