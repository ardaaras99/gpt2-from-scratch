import re

import tiktoken


class BPE:
    def __init__(self, name="gpt2"):
        self.tokenizer = tiktoken.get_encoding(name)

    def encode(self, text, allowed_special=None):
        if allowed_special is None:
            return self.tokenizer.encode(text)
        else:
            return self.tokenizer.encode(text, allowed_special=allowed_special)

    def decode(self, ids):
        return self.tokenizer.decode(ids)


class SimpleTokenizerV2:
    def __init__(self, full_text: str, remove_whitespace=True):
        self.full_text = full_text
        self.remove_whitespace = remove_whitespace
        self.str_to_int, self.vocab_size = self.create_vocab(full_text)
        self.int_to_str = {i: s for s, i in self.str_to_int.items()}

    def encode(self, text):
        text = self.process_raw_text(text)
        text = self.add_special_tokens(text)
        ids = [self.str_to_int[s] for s in text]
        return ids

    def add_special_tokens(self, processed_text):
        tmp = []
        for item in processed_text:
            if item not in self.str_to_int:
                tmp.append("<|unk|>")
            else:
                tmp.append(item)

        tmp.append("<|end|>")
        return tmp

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text

    def process_raw_text(self, text):
        temp = re.split(r'([,.?_!"()\']|--|\s)', text)
        if self.remove_whitespace:
            temp = [item.strip() for item in temp if item.strip()]
        return temp

    def create_vocab(self, text):
        processed_text = self.process_raw_text(text)
        all_words = sorted(set(processed_text))
        all_words.extend(["<|unk|>", "<|end|>"])
        vocab_size = len(all_words)
        vocab = {word: i for i, word in enumerate(all_words)}

        return vocab, vocab_size
