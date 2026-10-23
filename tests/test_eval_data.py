"""Sanity checks of the evaluation data and of the answer matcher."""

from pathlib import Path

import pytest

from scripts.evaluate import EVAL_SETS, load_cases, value_matches
from src.io_utils import load_functions
from src.prompt_builder import PROMPT_STYLES, build_prompt

PYTHON_TYPES: dict[str, tuple[type, ...]] = {
    "number": (int, float), "integer": (int,), "string": (str,),
    "boolean": (bool,),
}


@pytest.mark.parametrize("set_name", list(EVAL_SETS))
def test_cases_match_their_functions(set_name: str) -> None:
    """Every expected answer uses a real function, its exact params and
    values of the right type."""
    functions_path, cases_path = EVAL_SETS[set_name]
    functions = {f.name: f for f in load_functions(functions_path)}
    for case in load_cases(cases_path):
        function = functions[case.name]
        assert list(case.parameters) == list(function.parameters)
        for param, expected in case.parameters.items():
            options = expected if isinstance(expected, list) else [expected]
            allowed = PYTHON_TYPES[function.parameters[param].type]
            for option in options:
                assert isinstance(option, allowed), (case.prompt, param)
                if function.parameters[param].type != "boolean":
                    assert not isinstance(option, bool)


def test_basic_cases_follow_input_file() -> None:
    """The basic set has exactly the prompts of the example input."""
    from src.io_utils import load_prompts

    root = Path(__file__).resolve().parent.parent
    prompts = [p.prompt for p in
               load_prompts(root / "data/input/function_calling_tests.json")]
    assert [c.prompt for c in load_cases(EVAL_SETS["basic"][1])] == prompts


def test_value_matches() -> None:
    """Numbers compare as numbers, lists are alternatives, bools strict."""
    assert value_matches(2, 2.0)
    assert value_matches(["\\d+", "[0-9]+"], "[0-9]+")
    assert not value_matches("*", "asterisk")
    assert not value_matches(1, True)
    assert value_matches(False, False)


@pytest.mark.parametrize("style", PROMPT_STYLES)
def test_every_style_builds(style: str) -> None:
    """Each style produces a chat prompt ending at the assistant turn."""
    functions = load_functions(EVAL_SETS["basic"][0])
    text = build_prompt(functions, "Greet john", style)
    assert "fn_greet(name: string)" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_unknown_style() -> None:
    """A typo in the style name is an error, not a silent default."""
    with pytest.raises(ValueError):
        build_prompt([], "x", "fancy")
