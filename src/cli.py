"""Command line interface: argument parsing and top-level orchestration."""

import argparse
import sys
import time
from pathlib import Path

from src.constraints import TokenIndex
from src.generator import Generator
from src.io_utils import load_functions, load_prompts, write_calls
from src.llm import LanguageModel
from src.models import FunctionCall, FunctionDefinition, PromptItem
from src.vocab import load_vocabulary

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calls.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Arguments to parse; ``None`` means ``sys.argv[1:]``.

    Returns:
        The parsed namespace (functions_definition, input, output).
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural language prompts into function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
        help=f"function definitions file (default: {DEFAULT_FUNCTIONS})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"prompts file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def load_engine() -> tuple[LanguageModel, TokenIndex]:
    """Load the model and the token index the constrained decoding needs.

    Returns:
        The model and the token index.

    Raises:
        ValueError: If the model or its tokenizer files cannot be loaded.
    """
    print("Loading model...", file=sys.stderr)
    model = LanguageModel.load()
    size = len(model.logits(model.encode("hi")))
    vocab = load_vocabulary(model.vocab_path(), model.tokenizer_path(), size)
    return model, TokenIndex.build(vocab)


def build_generator(functions: list[FunctionDefinition]) -> Generator:
    """Load everything and return a ready Generator.

    Args:
        functions: Available functions.

    Returns:
        A ready Generator (default prompt style).

    Raises:
        ValueError: If the model or its tokenizer files cannot be loaded.
    """
    model, index = load_engine()
    return Generator(model=model, index=index, functions=functions)


def run_prompts(
    generator: Generator, prompts: list[PromptItem]
) -> list[FunctionCall]:
    """Generate one call per prompt; a failing prompt is reported, skipped.

    Args:
        generator: The ready generator.
        prompts: Requests to translate.

    Returns:
        The calls that could be generated, in input order.
    """
    calls: list[FunctionCall] = []
    total = len(prompts)
    for number, item in enumerate(prompts, start=1):
        start = time.perf_counter()
        try:
            call = generator.generate(item.prompt)
        except Exception as err:  # un prompt rate ne doit pas tout arreter
            print(
                f"[{number}/{total}] WARNING: skipped {item.prompt!r}: {err}",
                file=sys.stderr,
            )
            continue
        elapsed = time.perf_counter() - start
        print(
            f"[{number}/{total}] {elapsed:5.1f}s {call.name} "
            f"{call.parameters}",
            file=sys.stderr,
        )
        calls.append(call)
    return calls


def main(argv: list[str] | None = None) -> int:
    """Run the whole program and convert any failure into an exit code.

    Args:
        argv: Optional argument list (used by tests).

    Returns:
        0 on success, 1 on error, 130 if interrupted.
    """
    args = parse_args(argv)
    try:
        start = time.perf_counter()
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
        calls: list[FunctionCall] = []
        if prompts:  # pas de prompt : inutile de charger le modele
            generator = build_generator(functions)
            calls = run_prompts(generator, prompts)
            print(
                f"Model calls: {generator.stats.model_calls}, "
                f"forced bytes: {generator.stats.forced_bytes}",
                file=sys.stderr,
            )
        write_calls(args.output, calls)
        print(
            f"Wrote {len(calls)}/{len(prompts)} call(s) to {args.output} "
            f"in {time.perf_counter() - start:.1f}s"
        )
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0
