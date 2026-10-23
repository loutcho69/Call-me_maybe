"""A fake SDK model: byte-by-byte tokenizer and random logits."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.vocab import Vocabulary, bytes_to_unicode

SPECIALS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def tiny_vocabulary() -> tuple[Vocabulary, dict[str, int]]:
    """A byte-level vocabulary: the 256 bytes, a few words, 4 specials.

    Returns:
        The vocabulary and the special token ids by text.
    """
    tokens = [bytes([byte]) for byte in range(256)]
    tokens += [b"fn", b"_add", b"_numbers", b' "', b'",', b'"}', b"12",
               b"hello", b"true", b"fal"]
    specials = {text: len(tokens) + i for i, text in enumerate(SPECIALS)}
    tokens += [b""] * len(SPECIALS)
    vocab = Vocabulary(
        token_bytes=tokens, special_ids=frozenset(specials.values())
    )
    return vocab, specials


class FakeTensor(BaseModel):
    """Mimics the 2D tensor returned by the SDK's encode().

    Attributes:
        ids: Token ids.
    """

    ids: list[int]

    def tolist(self) -> list[list[int]]:
        """Return ``[[ids...]]`` like a torch tensor of shape (1, n)."""
        return [self.ids]


class FakeBackend(BaseModel):
    """Same public methods as ``Small_LLM_Model``; random logits.

    Attributes:
        vocab: Vocabulary to tokenize with.
        specials: Special token ids by text.
        seed: Random seed (each seed = a different "model").
        folder: Where vocab.json / tokenizer.json are written, if needed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vocab: Vocabulary
    specials: dict[str, int]
    seed: int = 0
    folder: Path | None = None

    def encode(self, text: str) -> FakeTensor:
        """Specials become one id each; any other byte its byte token."""
        single = {
            data: i for i, data in enumerate(self.vocab.token_bytes)
            if len(data) == 1
        }
        pattern = "(" + "|".join(re.escape(s) for s in SPECIALS) + ")"
        ids: list[int] = []
        for part in re.split(pattern, text):
            if part in self.specials:
                ids.append(self.specials[part])
            else:
                ids.extend(single[bytes([b])] for b in part.encode("utf-8"))
        return FakeTensor(ids=ids)

    def decode(self, ids: list[int]) -> str:
        """Concatenate the token bytes."""
        data = b"".join(self.vocab.token_bytes[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def get_logits_from_input_ids(self, input_ids: list[int]) -> Any:
        """Random scores, reproducible for a given seed and length."""
        rng = np.random.default_rng(self.seed * 1_000_003 + len(input_ids))
        return rng.normal(size=self.vocab.size)

    def get_path_to_vocab_file(self) -> str:
        """Write and return a vocab.json matching the vocabulary."""
        folder = self._folder()
        table = bytes_to_unicode()
        content = {
            "".join(table[b] for b in data): i
            for i, data in enumerate(self.vocab.token_bytes) if data
        }
        path = folder / "vocab.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        return str(path)

    def get_path_to_tokenizer_file(self) -> str:
        """Write and return a tokenizer.json with the special tokens."""
        folder = self._folder()
        added = [{"id": i, "content": t} for t, i in self.specials.items()]
        path = folder / "tokenizer.json"
        path.write_text(json.dumps({"added_tokens": added}), "utf-8")
        return str(path)

    def _folder(self) -> Path:
        """Return the folder for the tokenizer files."""
        if self.folder is None:
            raise ValueError("FakeBackend.folder is not set")
        self.folder.mkdir(parents=True, exist_ok=True)
        return self.folder
