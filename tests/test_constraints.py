"""Tests of the constraints, on a tiny vocabulary and on the real one."""

import json
import random
import re

import numpy as np
import pytest

from src.constraints import (
    BooleanConstraint,
    Constraint,
    NameConstraint,
    NumberConstraint,
    StringConstraint,
    TokenIndex,
    make_value_constraint,
)
from src.vocab import Vocabulary

TINY_TOKENS = [
    b'"', b"\\", b'",', b'"}', b"a", b"b", b"1", b"2", b".", b"-", b",",
    b"}", b" ", b"\n", b"fn", b"_add", b"_numbers", b"_greet", b"true",
    b"fal", b"se", b"\\d", b"d", b"hello world", b'"x', b"",
]


@pytest.fixture
def tiny() -> TokenIndex:
    """Index over a handful of hand-written tokens (last one forbidden)."""
    vocab = Vocabulary(token_bytes=TINY_TOKENS, special_ids=frozenset())
    return TokenIndex.build(vocab)


def allowed_tokens(constraint: Constraint, index: TokenIndex) -> set[bytes]:
    """Return the bytes of every token the mask allows."""
    return {
        index.token_bytes[i] for i in np.flatnonzero(constraint.mask(index))
    }


# ---------------------------------------------------------------- index


def test_index_categories(tiny: TokenIndex) -> None:
    """Tokens land in the right precomputed categories."""
    plain = {TINY_TOKENS[i] for i in np.flatnonzero(tiny.plain_text)}
    assert b"hello world" in plain
    assert b'"' not in plain and b"\n" not in plain
    numeric = {TINY_TOKENS[i] for i in tiny.numeric_ids}
    assert numeric == {b"1", b"2", b".", b"-"}
    assert tiny.by_bytes[b"fn"] == TINY_TOKENS.index(b"fn")
    assert b"" not in tiny.by_bytes


# ---------------------------------------------------------------- names


def test_name_unique_completion(tiny: TokenIndex) -> None:
    """Once only one name fits, the rest is forced without the model."""
    name = NameConstraint(names=[b"fn_add_numbers", b"fn_greet"])
    assert allowed_tokens(name, tiny) == {b"fn"}
    assert not name.allows(b"_greet")
    name.advance(b"fn")
    assert name.forced_completion() is None  # two names still possible
    assert allowed_tokens(name, tiny) == {b"_add", b"_greet"}
    name.advance(b"_add")
    forced = name.forced_completion()
    assert forced == b"_numbers"
    name.advance(forced)
    assert name.done and name.value() == "fn_add_numbers"


def test_name_ambiguous_prefix(tiny: TokenIndex) -> None:
    """``fn_add`` vs ``fn_add_numbers``: a quote closes the short one."""
    name = NameConstraint(names=[b"fn_add", b"fn_add_numbers"])
    name.advance(b"fn")
    name.advance(b"_add")
    assert not name.done
    assert {b'"', b'",', b'"}', b"_numbers"} <= allowed_tokens(name, tiny)
    assert not name.allows(b'"x')
    name.advance(b'",')
    assert name.done and name.value() == "fn_add"


# ---------------------------------------------------------------- numbers


def test_number_negative_decimal() -> None:
    """``-1.2`` then a comma gives -1.2."""
    number = NumberConstraint()
    for token in (b"-", b"1", b".", b"2"):
        number.advance(token)
    assert not number.allows(b".")
    assert not number.allows(b"-")
    number.advance(b",")
    assert number.done and number.value() == -1.2


def test_number_needs_a_digit(tiny: TokenIndex) -> None:
    """No terminator before the first digit."""
    number = NumberConstraint()
    assert not number.allows(b",")
    number.advance(b"-")
    assert not number.allows(b"}")
    assert allowed_tokens(number, tiny) == {b"1", b"2"}


def test_number_is_float() -> None:
    """A ``number`` is always a float in the output: 2 -> 2.0."""
    number = NumberConstraint()
    number.advance(b"2")
    number.advance(b"}")
    value = number.value()
    assert isinstance(value, float) and value == 2.0


def test_integer_rejects_dot() -> None:
    """The ``integer`` type has no decimal point."""
    integer = NumberConstraint(integer=True)
    integer.advance(b"1")
    assert not integer.allows(b".")
    integer.advance(b" ")
    assert integer.value() == 1 and isinstance(integer.value(), int)


def test_number_max_length() -> None:
    """Past the limit only a terminator is allowed."""
    number = NumberConstraint(max_length=3)
    for token in (b"1", b"2", b"3"):
        number.advance(token)
    assert not number.allows(b"4")
    assert number.allows(b",")


# ---------------------------------------------------------------- strings


def test_string_simple_close() -> None:
    """Text then a closing quote."""
    string = StringConstraint()
    string.advance(b"hello world")
    string.advance(b'"}')
    assert string.done and string.value() == "hello world"


def test_string_escaped_quote() -> None:
    """A quote inside the text must be written ``\\"``."""
    string = StringConstraint()
    string.advance(b"\\")
    assert string.escape
    string.advance(b'"')  # escaped: does not close
    assert not string.done
    string.advance(b"a")
    string.advance(b'"')
    assert string.value() == '"a'


def test_string_regex_backslash() -> None:
    r"""The regex ``\d`` must be written ``\\d``; ``\d`` is invalid JSON."""
    raw = StringConstraint()
    assert not raw.allows(b"\\d")
    good = StringConstraint()
    good.advance(b"\\")
    good.advance(b"\\d")
    good.advance(b'"')
    assert good.value() == "\\d"


def test_string_rejects_control_and_bad_tail() -> None:
    """Raw newline inside a string, or text after the close: refused."""
    string = StringConstraint()
    assert not string.allows(b"\n")
    assert not string.allows(b'"x')
    assert string.allows(b'",')


def test_string_max_bytes(tiny: TokenIndex) -> None:
    """When full, only closing tokens remain."""
    string = StringConstraint(max_bytes=2)
    string.advance(b"a")
    string.advance(b"b")
    assert allowed_tokens(string, tiny) == {b'"', b'",', b'"}'}
    assert not string.allows(b"\\")


def test_string_utf8() -> None:
    """Accents written byte by byte come back as text."""
    string = StringConstraint()
    for byte in "é".encode("utf-8"):
        string.advance(bytes([byte]))
    string.advance(b'"')
    assert string.value() == "é"


# ---------------------------------------------------------------- booleans


def test_boolean(tiny: TokenIndex) -> None:
    """``fal`` + ``se`` gives False; anything else is refused."""
    boolean = BooleanConstraint()
    assert allowed_tokens(boolean, tiny) == {b"true", b"fal"}
    boolean.advance(b"fal")
    assert not boolean.allows(b"x")
    boolean.advance(b"se")
    assert boolean.done and boolean.value() is False


def test_factory() -> None:
    """Each type gets its constraint."""
    assert isinstance(make_value_constraint("string"), StringConstraint)
    assert isinstance(make_value_constraint("boolean"), BooleanConstraint)
    integer = make_value_constraint("integer")
    assert isinstance(integer, NumberConstraint) and integer.integer


# ------------------------------------------------- real vocabulary, random


def random_walk(
    constraint: Constraint, index: TokenIndex, rng: random.Random,
    favour: set[int] | None = None,
) -> Constraint:
    """Pick random allowed tokens until the value is done.

    Args:
        constraint: Fresh constraint to drive.
        index: Real token index.
        rng: Random generator (seeded for reproducible tests).
        favour: Ids picked more often when allowed (to reach rare paths).

    Returns:
        The finished constraint.
    """
    for _ in range(400):
        if constraint.done:
            return constraint
        forced = constraint.forced_completion()
        if forced is not None and rng.random() < 0.5:
            constraint.advance(forced)
            continue
        allowed = np.flatnonzero(constraint.mask(index)).tolist()
        assert allowed, "dead end: no token allowed"
        preferred = [i for i in allowed if favour and i in favour]
        pool = preferred if preferred and rng.random() < 0.3 else allowed
        constraint.advance(index.token_bytes[rng.choice(pool)])
    raise AssertionError("value never finished")


def test_random_strings_are_valid_json(real_index: TokenIndex) -> None:
    """Random strings always decode and survive a JSON round trip."""
    rng = random.Random(0)
    tricky = set(real_index.string_tricky_ids)
    for _ in range(40):
        string = random_walk(StringConstraint(), real_index, rng, tricky)
        assert isinstance(string, StringConstraint)
        value = string.value()
        assert json.loads(json.dumps(value)) == value


def test_random_numbers(real_index: TokenIndex) -> None:
    """Random numbers are floats; integers are ints."""
    rng = random.Random(1)
    ends = set(np.flatnonzero(real_index.ends_value).tolist())
    for integer in (False, True):
        for _ in range(100):
            number = random_walk(
                NumberConstraint(integer=integer), real_index, rng, ends
            )
            assert isinstance(number, NumberConstraint)
            assert re.fullmatch(r"-?\d+(\.\d*)?", number.text)
            assert isinstance(number.value(), int if integer else float)


def test_random_names(real_index: TokenIndex) -> None:
    """Random names are always one of the known names."""
    rng = random.Random(2)
    names = [
        b"fn_add_numbers", b"fn_add", b"fn_greet", b"fn_reverse_string",
        b"fn_get_square_root", b"fn_substitute_string_with_regex",
    ]
    tricky = set(real_index.string_tricky_ids)
    seen = set()
    for _ in range(200):
        fresh = NameConstraint(names=names)
        name = random_walk(fresh, real_index, rng, tricky)
        assert isinstance(name, NameConstraint)
        assert name.content in names
        seen.add(name.content)
    assert b"fn_add" in seen  # the ambiguous short name is reachable


def test_random_booleans(real_index: TokenIndex) -> None:
    """Random booleans are True or False."""
    rng = random.Random(3)
    values = {
        random_walk(BooleanConstraint(), real_index, rng).value()
        for _ in range(50)
    }
    assert values == {True, False}
