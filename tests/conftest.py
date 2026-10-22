"""Shared test fixtures."""

import os
from pathlib import Path

import pytest

from src.constraints import TokenIndex
from src.vocab import load_vocabulary

# Taille des logits de Qwen3-0.6B (mesuree en phase 2).
QWEN_LOGITS_SIZE = 151936


def find_tokenizer_dir() -> Path | None:
    """Locate vocab.json + tokenizer.json of Qwen3-0.6B, if available.

    Looks at ``$QWEN_TOKENIZER_DIR`` first, then the Hugging Face cache.

    Returns:
        The folder holding both files, or None.
    """
    candidates: list[Path] = []
    env = os.environ.get("QWEN_TOKENIZER_DIR")
    if env:
        candidates.append(Path(env))
    hf_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    snapshots = hf_home / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots"
    candidates.extend(sorted(snapshots.glob("*")))
    for folder in candidates:
        if (folder / "vocab.json").is_file() and (
            folder / "tokenizer.json"
        ).is_file():
            return folder
    return None


@pytest.fixture(scope="session")
def real_index() -> TokenIndex:
    """TokenIndex built from the real Qwen vocabulary (or skip)."""
    folder = find_tokenizer_dir()
    if folder is None:
        pytest.skip("Qwen tokenizer files not found")
    vocab = load_vocabulary(
        folder / "vocab.json", folder / "tokenizer.json", QWEN_LOGITS_SIZE
    )
    return TokenIndex.build(vocab)
