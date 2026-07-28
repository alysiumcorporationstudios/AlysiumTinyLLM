import json
import re

class CharTokenizer:
    """
    Simple character-level tokenizer.
    Builds vocabulary from the training text.
    """

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

    def build_vocab(self, text: str):
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        return self

    def encode(self, text: str):
        return [self.char_to_id.get(ch, 0) for ch in text]

    def decode(self, ids):
        return "".join(self.id_to_char.get(i, "?") for i in ids)

    def save(self, path="tokenizer.json"):
        data = {
            "char_to_id": self.char_to_id,
            "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path="tokenizer.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        self.vocab_size = data["vocab_size"]
        return self
