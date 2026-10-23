"""Measure accuracy of each prompt style on cases with known answers.

Run with: ``uv run python -m scripts.evaluate`` (all styles, all sets)
or e.g. ``uv run python -m scripts.evaluate --styles rules --sets extra``.

In a case file, an expected value written as a list means "any of
these values is accepted" (useful for regexes like ``\\d+`` / ``[0-9]+``).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

from src.cli import load_engine
from src.constraints import TokenIndex
from src.generator import Generator
from src.io_utils import load_functions
from src.llm import LanguageModel
from src.models import FunctionCall
from src.prompt_builder import PROMPT_STYLES

ROOT = Path(__file__).resolve().parent.parent
EVAL_SETS = {
    "basic": (ROOT / "data/input/functions_definition.json",
              ROOT / "tests/eval/basic_cases.json"),
    "extra": (ROOT / "tests/eval/extra_functions.json",
              ROOT / "tests/eval/extra_cases.json"),
}


class EvalCase(BaseModel):
    """A prompt with its expected answer.

    Attributes:
        prompt: The request.
        name: Expected function name.
        parameters: Expected values (a list = accepted alternatives).
    """

    prompt: str
    name: str
    parameters: dict[str, Any]


class SetResult(BaseModel):
    """Score of one prompt style on one case set.

    Attributes:
        correct: Number of exact answers.
        total: Number of cases.
        seconds: Time spent.
    """

    correct: int = 0
    total: int = 0
    seconds: float = 0.0


def load_cases(path: Path) -> list[EvalCase]:
    """Read a case file.

    Args:
        path: JSON file with a list of cases.

    Returns:
        The cases.
    """
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return TypeAdapter(list[EvalCase]).validate_python(raw)


def value_matches(expected: Any, got: Any) -> bool:
    """Compare one generated value with the expected one.

    Args:
        expected: Expected value, or list of accepted values.
        got: Generated value.

    Returns:
        True if they match (numbers compared as numbers).
    """
    if isinstance(expected, list):
        return any(value_matches(option, got) for option in expected)
    if isinstance(expected, bool) or isinstance(got, bool):
        return expected is got
    if isinstance(expected, (int, float)):
        return isinstance(got, (int, float)) and abs(got - expected) < 1e-9
    return bool(expected == got)


def problems(case: EvalCase, call: FunctionCall) -> list[str]:
    """List the differences between a call and the expected answer.

    Args:
        case: The expected answer.
        call: The generated call.

    Returns:
        Human readable differences (empty if the call is right).
    """
    if call.name != case.name:
        return [f"name {call.name} (expected {case.name})"]
    found = []
    for param, expected in case.parameters.items():
        got = call.parameters.get(param)
        if not value_matches(expected, got):
            found.append(f"{param}={got!r} (expected {expected!r})")
    return found


def run_set(
    model: LanguageModel, index: TokenIndex, style: str, set_name: str
) -> SetResult:
    """Run one prompt style on one case set, printing each failure.

    Args:
        model: Loaded model.
        index: Token index.
        style: Prompt style.
        set_name: Key of EVAL_SETS.

    Returns:
        The score.
    """
    functions_path, cases_path = EVAL_SETS[set_name]
    generator = Generator(model=model, index=index,
                          functions=load_functions(functions_path),
                          prompt_style=style)
    cases = load_cases(cases_path)
    result = SetResult(total=len(cases))
    start = time.perf_counter()
    for case in cases:
        try:
            issues = problems(case, generator.generate(case.prompt))
        except Exception as err:  # on note l'echec et on continue
            issues = [f"crash: {err}"]
        if issues:
            print(f"  FAIL [{style}/{set_name}] {case.prompt!r}: "
                  + "; ".join(issues))
        else:
            result.correct += 1
    result.seconds = time.perf_counter() - start
    return result


def main() -> int:
    """Evaluate the requested styles on the requested sets.

    Returns:
        0 when the evaluation ran, 1 on loading errors.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--styles", default=",".join(PROMPT_STYLES))
    parser.add_argument("--sets", default=",".join(EVAL_SETS))
    args = parser.parse_args()
    styles = args.styles.split(",")
    sets = args.sets.split(",")
    unknown = [s for s in styles if s not in PROMPT_STYLES]
    unknown += [s for s in sets if s not in EVAL_SETS]
    if unknown:
        print(f"Error: unknown style or set: {unknown}", file=sys.stderr)
        return 1
    try:
        model, index = load_engine()
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    table: dict[str, dict[str, SetResult]] = {}
    for style in styles:
        print(f"== style {style}")
        table[style] = {name: run_set(model, index, style, name)
                        for name in sets}

    print("\nstyle      " + "".join(f"{name:>16}" for name in sets)
          + "      total")
    for style, results in table.items():
        cells = "".join(
            f"{r.correct:>4}/{r.total:<3}{r.seconds:>6.1f}s  "
            for r in results.values()
        )
        correct = sum(r.correct for r in results.values())
        total = sum(r.total for r in results.values())
        print(f"{style:<10} {cells}  {100 * correct / total:5.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
