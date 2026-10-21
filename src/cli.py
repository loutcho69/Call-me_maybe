"""Command line interface: argument parsing and top-level orchestration."""

import argparse
import sys
from pathlib import Path

from src.io_utils import load_functions, load_prompts

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


def main(argv: list[str] | None = None) -> int:
    """Run the whole program and convert any failure into an exit code.

    Args:
        argv: Optional argument list (used by tests).

    Returns:
        0 on success, 1 on error, 130 if interrupted.
    """
    args = parse_args(argv)
    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
        print(
            f"Loaded {len(functions)} function(s) "
            f"and {len(prompts)} prompt(s)."
        )
        # Etape suivante : charger le modele et generer les appels.
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0
