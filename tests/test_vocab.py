"""Tests of the byte-level vocabulary, with a tiny fake vocabulary."""

import json
from pathlib import Path

import pytest

from src.vocab import bytes_to_unicode, load_vocabulary, token_to_bytes


def test_table_covers_every_byte_once() -> None:
    """The 256 bytes map to 256 different characters."""
    table = bytes_to_unicode()
    assert len(table) == 256
    assert len(set(table.values())) == 256


def test_known_characters() -> None:
    """Space and newline get their well-known GPT-2 characters."""
    table = bytes_to_unicode()
    assert table[ord(" ")] == "Ġ"
    assert table[ord("\n")] == "Ċ"
    assert table[ord("a")] == "a"


def test_token_to_bytes() -> None:
    """Tokens from vocab.json decode to their real bytes."""
    assert token_to_bytes("Ġsum") == b" sum"
    assert token_to_bytes("Ã©") == "é".encode("utf-8")
    assert token_to_bytes("Ã") == b"\xc3"  # half of an accented letter
    assert token_to_bytes("€") is None  # not in the byte-level table


def _write(path: Path, data: object) -> Path:
    """Write data as JSON to a file and return its path."""
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_vocabulary(tmp_path: Path) -> None:
    """Regular, special and missing ids are handled."""
    vocab = _write(tmp_path / "vocab.json", {"a": 0, "Ġb": 1, "Ã©": 2})
    tokenizer = _write(
        tmp_path / "tokenizer.json",
        {"added_tokens": [{"id": 3, "content": "<|im_end|>"}]},
    )
    result = load_vocabulary(vocab, tokenizer, size=5)
    assert result.size == 5
    assert result.token_bytes == [b"a", b" b", b"\xc3\xa9", b"", b""]
    assert result.special_ids == frozenset({3})
    assert result.usable_mask().tolist() == [True, True, True, False, False]


def test_missing_file(tmp_path: Path) -> None:
    """A missing file gives a clear ValueError."""
    with pytest.raises(ValueError, match="cannot read tokenizer file"):
        load_vocabulary(tmp_path / "nope.json", tmp_path / "x.json", 3)


def test_wrong_shape(tmp_path: Path) -> None:
    """vocab.json must be an object."""
    vocab = _write(tmp_path / "vocab.json", ["a", "b"])
    tokenizer = _write(tmp_path / "tokenizer.json", {})
    with pytest.raises(ValueError, match="JSON object"):
        load_vocabulary(vocab, tokenizer, 3)
