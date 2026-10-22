"""Phase 2 check on the real model: vocabulary decoding and timings.

Run with: ``uv run python -m scripts.check_vocab``
"""

import random
import sys
import time

from src.llm import LanguageModel
from src.vocab import Vocabulary, load_vocabulary


def check_decoding(model: LanguageModel, vocab: Vocabulary) -> bool:
    """Compare our bytes with the SDK's decode() on 500 random tokens.

    Tokens whose bytes are not complete UTF-8 (half of an accented
    letter) are skipped: decode() shows them as a replacement character.

    Args:
        model: The loaded model.
        vocab: Our vocabulary.

    Returns:
        True if every compared token matches.
    """
    usable = [i for i in range(vocab.size) if vocab.token_bytes[i]]
    sample = random.Random(42).sample(usable, min(500, len(usable)))
    compared, mismatches = 0, []
    for token_id in sample:
        try:
            ours = vocab.token_bytes[token_id].decode("utf-8")
        except UnicodeDecodeError:
            continue
        compared += 1
        theirs = model.decode([token_id])
        if ours != theirs:
            mismatches.append((token_id, ours, theirs))
    print(f"decoding: {compared - len(mismatches)}/{compared} identical")
    for token_id, ours, theirs in mismatches[:5]:
        print(f"  id {token_id}: ours={ours!r} sdk={theirs!r}")
    return not mismatches


def check_special(model: LanguageModel, vocab: Vocabulary) -> bool:
    """Check that chat markers are single special tokens.

    Args:
        model: The loaded model.
        vocab: Our vocabulary.

    Returns:
        True if each marker is one special id.
    """
    ok = True
    for marker in ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]:
        ids = model.encode(marker)
        single = len(ids) == 1 and ids[0] in vocab.special_ids
        ok = ok and single
        print(f"{marker!r:16} -> {ids} {'OK' if single else 'PROBLEM'}")
    return ok


def show_example(model: LanguageModel, vocab: Vocabulary) -> None:
    """Print how a sentence is split into tokens.

    Args:
        model: The loaded model.
        vocab: Our vocabulary.
    """
    text = 'What is the sum of 265 and 345? {"name": "fn_add_numbers"'
    print(f"tokens of {text!r}:")
    for token_id in model.encode(text):
        print(f"  {token_id:>7} {vocab.token_bytes[token_id]!r}")


def measure_speed(model: LanguageModel) -> None:
    """Time one logits call for several sequence lengths.

    Args:
        model: The loaded model.
    """
    base = model.encode("The quick brown fox jumps over the lazy dog. ")
    for length in (100, 300, 600):
        ids = (base * (length // len(base) + 1))[:length]
        start = time.perf_counter()
        for _ in range(3):
            model.logits(ids)
        elapsed = (time.perf_counter() - start) / 3
        print(f"logits for {length:>3} tokens: {elapsed * 1000:7.1f} ms")


def main() -> int:
    """Run all phase 2 checks.

    Returns:
        0 if all checks pass, 1 otherwise.
    """
    try:
        model = LanguageModel.load()
        size = len(model.logits(model.encode("hi")))
        start = time.perf_counter()
        vocab = load_vocabulary(
            model.vocab_path(), model.tokenizer_path(), size
        )
        load_ms = (time.perf_counter() - start) * 1000
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    usable = int(vocab.usable_mask().sum())
    print(f"logits size: {size}, usable tokens: {usable}, "
          f"special: {len(vocab.special_ids)}, load: {load_ms:.0f} ms")
    ok = check_decoding(model, vocab)
    ok = check_special(model, vocab) and ok
    show_example(model, vocab)
    measure_speed(model)
    print("ALL CHECKS OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
