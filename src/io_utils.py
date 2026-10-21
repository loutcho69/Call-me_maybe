"""Reading, validating and writing the JSON files of the project."""

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from src.models import FunctionCall, FunctionDefinition, PromptItem

_FUNCTIONS_ADAPTER: TypeAdapter[list[FunctionDefinition]] = TypeAdapter(
    list[FunctionDefinition]
)
_PROMPTS_ADAPTER: TypeAdapter[list[PromptItem]] = TypeAdapter(
    list[PromptItem]
)


def _read_json(path: Path) -> Any:
    """Load a JSON file, turning every failure into a clear ValueError.

    Args:
        path: File to read.

    Returns:
        The decoded JSON value.

    Raises:
        ValueError: If the file is missing, unreadable or not valid JSON.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise ValueError(f"file not found: {path}") from None
    except IsADirectoryError:
        raise ValueError(f"expected a file, got a directory: {path}") from None
    except UnicodeDecodeError:
        raise ValueError(f"file is not valid UTF-8: {path}") from None
    except json.JSONDecodeError as err:
        raise ValueError(
            f"invalid JSON in {path} (line {err.lineno}, "
            f"column {err.colno}): {err.msg}"
        ) from None
    except OSError as err:
        raise ValueError(f"cannot read {path}: {err.strerror}") from None


def _format_validation_error(path: Path, err: ValidationError) -> str:
    """Build a short, readable message from a pydantic ValidationError.

    Args:
        path: File that failed validation.
        err: The pydantic error.

    Returns:
        A one-line-per-problem message.
    """
    lines = [f"invalid content in {path}:"]
    for issue in err.errors()[:5]:
        location = ".".join(str(part) for part in issue["loc"]) or "<root>"
        lines.append(f"  - at {location}: {issue['msg']}")
    if err.error_count() > 5:
        lines.append(f"  ... and {err.error_count() - 5} more")
    return "\n".join(lines)


def load_functions(path: Path) -> list[FunctionDefinition]:
    """Load and validate the function definitions file.

    Args:
        path: Path to functions_definition.json.

    Returns:
        The validated list of functions.

    Raises:
        ValueError: On any reading, JSON or schema problem.
    """
    raw = _read_json(path)
    try:
        functions = _FUNCTIONS_ADAPTER.validate_python(raw)
    except ValidationError as err:
        raise ValueError(_format_validation_error(path, err)) from None
    if not functions:
        raise ValueError(f"no function defined in {path}")
    names = [function.name for function in functions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate function names in {path}: {', '.join(duplicates)}"
        )
    return functions


def load_prompts(path: Path) -> list[PromptItem]:
    """Load and validate the prompts file.

    Args:
        path: Path to function_calling_tests.json.

    Returns:
        The validated list of prompts (possibly empty).

    Raises:
        ValueError: On any reading, JSON or schema problem.
    """
    raw = _read_json(path)
    try:
        return _PROMPTS_ADAPTER.validate_python(raw)
    except ValidationError as err:
        raise ValueError(_format_validation_error(path, err)) from None


def write_calls(path: Path, calls: list[FunctionCall]) -> None:
    """Write the results as a JSON array, creating parent folders.

    Args:
        path: Destination file.
        calls: Results to serialize.

    Raises:
        ValueError: If the file cannot be written.
    """
    payload = [call.model_dump() for call in calls]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except OSError as err:
        raise ValueError(f"cannot write {path}: {err.strerror}") from None
