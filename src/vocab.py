"""Vocabulary: the real bytes written by each token id.

Qwen uses byte-level BPE, like GPT-2. In ``vocab.json`` a token is not
stored as plain text: every byte is replaced by a printable unicode
character (a space becomes ``Ġ``, a newline ``Ċ``...). We invert that
table to know exactly which bytes each token produces.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict


def bytes_to_unicode() -> dict[int, str]:
    """Build the GPT-2 table: byte value -> printable character.

    Printable bytes keep their own character. The other ones (control
    characters, space...) are shifted to code points 256 and above, in
    order, so that every byte gets a visible, unique character.

    Returns:
        A dict with the 256 byte values as keys.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    table = {byte: chr(byte) for byte in printable}
    shift = 0
    for byte in range(256):
        if byte not in table:
            table[byte] = chr(256 + shift)
            shift += 1
    return table


# Table inverse, calculee une seule fois : caractere -> octet.
_CHAR_TO_BYTE: dict[str, int] = {
    char: byte for byte, char in bytes_to_unicode().items()
}


def token_to_bytes(token: str) -> bytes | None:
    """Convert a token as written in vocab.json to its real bytes.

    Args:
        token: Token string, e.g. ``"Ġsum"``.

    Returns:
        The bytes (``b" sum"``), or None if the string contains a
        character outside the byte-level table.
    """
    try:
        return bytes(_CHAR_TO_BYTE[char] for char in token)
    except KeyError:
        return None


class Vocabulary(BaseModel):
    """Bytes of every token id the model can output.

    Attributes:
        token_bytes: ``token_bytes[i]`` = bytes written by token ``i``;
            ``b""`` for ids we must never generate.
        special_ids: Ids of special tokens (``<|im_end|>``, ``<think>``...).
    """

    model_config = ConfigDict(frozen=True)

    token_bytes: list[bytes]
    special_ids: frozenset[int]

    @property
    def size(self) -> int:
        """Number of token ids (equals the length of the logits)."""
        return len(self.token_bytes)

    def usable_mask(self) -> npt.NDArray[np.bool_]:
        """Mask of the ids that write real text.

        Returns:
            Boolean array of length ``size``: True if the token may be
            generated inside a value, False for special or unknown ids.
        """
        return np.array(
            [len(data) > 0 for data in self.token_bytes], dtype=np.bool_
        )


def _read_json_file(path: Path) -> Any:
    """Load a JSON file, raising ValueError with a clear message.

    Args:
        path: File to read.

    Returns:
        The decoded JSON value.

    Raises:
        ValueError: If the file cannot be read or parsed.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ValueError(f"cannot read tokenizer file {path}: {err}") from None


def load_vocabulary(
    vocab_path: Path, tokenizer_path: Path, size: int
) -> Vocabulary:
    """Build the Vocabulary from the model files.

    Args:
        vocab_path: Path to vocab.json (``{token_string: id}``).
        tokenizer_path: Path to tokenizer.json (for ``added_tokens``).
        size: Number of logits returned by the model. It is a bit larger
            than vocab.json: the extra ids stay unusable.

    Returns:
        The Vocabulary.

    Raises:
        ValueError: If a file is missing or has an unexpected shape.
    """
    vocab = _read_json_file(vocab_path)
    if not isinstance(vocab, dict):
        raise ValueError(f"{vocab_path} should contain a JSON object")

    token_bytes: list[bytes] = [b""] * size
    for token, token_id in vocab.items():
        if not isinstance(token_id, int) or not 0 <= token_id < size:
            continue
        data = token_to_bytes(token)
        if data is not None:
            token_bytes[token_id] = data

    tokenizer = _read_json_file(tokenizer_path)
    added = tokenizer.get("added_tokens", []) if isinstance(
        tokenizer, dict
    ) else []
    special_ids: set[int] = set()
    for entry in added:
        token_id = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(token_id, int) and 0 <= token_id < size:
            special_ids.add(token_id)
            # Un token special n'ecrit jamais de texte dans une valeur.
            token_bytes[token_id] = b""

    return Vocabulary(
        token_bytes=token_bytes, special_ids=frozenset(special_ids)
    )
