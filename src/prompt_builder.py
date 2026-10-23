"""Build the text given to the model before it writes the JSON call.

The structure of the answer is enforced by constrained decoding, not by
this prompt. The prompt only helps the model pick the right function
and the right argument values.

Several prompt styles are kept so they can be measured against each
other (see ``scripts/evaluate.py``); ``DEFAULT_STYLE`` is the one used
by the program.
"""

from src.models import FunctionDefinition

PROMPT_STYLES = ("basic", "rules", "examples")
DEFAULT_STYLE = "rules"

_RULES = (
    "Rules:\n"
    "- Choose the function whose description matches what the user "
    "wants.\n"
    "- Copy text values exactly as written in the request, without the "
    "quotes around them.\n"
    "- Write numbers with digits.\n"
    "- When the request names a character in words (asterisks, dashes, "
    "underscores, spaces, dots...), use the character itself "
    '(*, -, _, " ", .).\n'
    "- A regex parameter is a Python regular expression. When it is "
    "about letters, include both lowercase and uppercase (for example "
    "[a-zA-Z] for any letter).\n"
)

_EXAMPLES = (
    "Examples (they use other functions, only the format matters):\n"
    "Request: Replace every comma in 'a,b,c' with semicolons\n"
    'Answer: {"name": "fn_replace", "parameters": {"text": "a,b,c", '
    '"pattern": ",", "replacement": ";"}}\n'
    "Request: Replace all letters in 'x1y2' with hashes\n"
    'Answer: {"name": "fn_replace", "parameters": {"text": "x1y2", '
    '"pattern": "[a-zA-Z]", "replacement": "#"}}\n'
    "Request: Subtract 4 from -2.5\n"
    'Answer: {"name": "fn_subtract", "parameters": {"x": -2.5, "y": 4.0}}\n'
)


def describe_function(function: FunctionDefinition) -> str:
    """One compact line per function, e.g. ``fn_add(a: number): Adds``.

    Parameter descriptions, when the definition has some, are added
    in parentheses after the type.

    Args:
        function: The function definition.

    Returns:
        The description line.
    """
    params = []
    for name, spec in function.parameters.items():
        detail = f" ({spec.description})" if spec.description else ""
        params.append(f"{name}: {spec.type}{detail}")
    return f"- {function.name}({', '.join(params)}): {function.description}"


def build_prompt(
    functions: list[FunctionDefinition],
    request: str,
    style: str = DEFAULT_STYLE,
) -> str:
    """Build the full chat text for Qwen3, ready to be encoded.

    The empty ``<think>`` block switches off Qwen3's reasoning mode:
    the model starts writing the answer at once.

    Args:
        functions: Every available function.
        request: The user's natural language request.
        style: One of ``PROMPT_STYLES``.

    Returns:
        The prompt text, ending where the JSON answer must start.

    Raises:
        ValueError: If the style is unknown.
    """
    if style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style {style!r}")
    catalog = "\n".join(describe_function(f) for f in functions)
    system = (
        "You translate a user request into one function call.\n"
        f"Available functions:\n{catalog}\n"
    )
    if style == "basic":
        system += (
            "Pick the function that does what the user asks, and take the "
            "argument values from the request.\n"
        )
    else:
        system += _RULES
    if style == "examples":
        system += _EXAMPLES
    system += (
        'Answer with JSON only: {"name": "<function>", "parameters": {...}}'
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{request}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
