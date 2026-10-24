"""Robustness: every bad input gives a clear message and exit code 1."""

import json
from pathlib import Path

import pytest

from src import cli
from src.generator import Generator
from src.io_utils import load_functions, load_prompts, write_calls
from src.llm import LanguageModel
from src.models import FunctionCall

ROOT = Path(__file__).resolve().parent.parent
GOOD_FUNCTIONS = ROOT / "data/input/functions_definition.json"
GOOD_PROMPTS = ROOT / "data/input/function_calling_tests.json"


def write(path: Path, content: str | bytes) -> Path:
    """Write text or bytes to a file and return its path."""
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


# ------------------------------------------------------------- reading


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "invalid JSON"),
        ("[{]", "invalid JSON"),
        (b"\xff\xfe\x00", "not valid UTF-8"),
        ('{"prompt": "x"}', "invalid content"),
        ('[{"text": "x"}]', "invalid content"),
        ('[{"prompt": 42}]', "invalid content"),
    ],
)
def test_bad_prompt_files(
    tmp_path: Path, content: str | bytes, message: str
) -> None:
    """Empty, broken, non UTF-8 or wrongly shaped prompt files."""
    path = write(tmp_path / "prompts.json", content)
    with pytest.raises(ValueError, match=message):
        load_prompts(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[]", "no function defined"),
        ('[{"name": "f", "parameters": {"a": {"type": "date"}}}]',
         "no usable function"),
        ('[{"name": "f", "parameters": {"a": {"kind": "x"}}}]',
         "invalid content"),
        ('[{"name": ""}]', "invalid content"),
        ('[{"name": "f"}, {"name": "f"}]', "duplicate function names"),
    ],
)
def test_bad_function_files(
    tmp_path: Path, content: str, message: str
) -> None:
    """Empty list, unknown type, empty name, duplicate names."""
    path = write(tmp_path / "functions.json", content)
    with pytest.raises(ValueError, match=message):
        load_functions(path)


def test_flexible_function_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unusual return types and type synonyms are accepted; a function
    with an unsupported parameter is skipped, the others are kept."""
    content = json.dumps([
        {"name": "fn_split", "parameters": {"s": {"type": "String"}},
         "returns": {"type": "array"}},
        {"name": "fn_half", "parameters": {"x": {"type": "float"},
                                           "n": {"type": "int"},
                                           "b": {"type": "bool"}},
         "returns": {"type": "null"}},
        {"name": "fn_sum_list", "parameters": {"xs": {"type": "array"}}},
    ])
    functions = load_functions(write(tmp_path / "f.json", content))
    assert [f.name for f in functions] == ["fn_split", "fn_half"]
    assert functions[0].parameters["s"].type == "string"
    assert [s.type for s in functions[1].parameters.values()] == [
        "number", "integer", "boolean"]
    assert "skipping fn_sum_list" in capsys.readouterr().err


def test_missing_and_directory(tmp_path: Path) -> None:
    """A missing path or a folder instead of a file."""
    with pytest.raises(ValueError, match="file not found"):
        load_prompts(tmp_path / "nope.json")
    with pytest.raises(ValueError, match="got a directory"):
        load_prompts(tmp_path)


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    """Extra keys in the inputs are tolerated (forward compatible)."""
    path = write(tmp_path / "p.json", '[{"prompt": "hi", "id": 3}]')
    assert [p.prompt for p in load_prompts(path)] == ["hi"]


# ------------------------------------------------------------- writing


def test_output_is_a_directory(tmp_path: Path) -> None:
    """Writing onto a folder fails cleanly."""
    with pytest.raises(ValueError, match="cannot write"):
        write_calls(tmp_path, [])


def test_output_parent_is_a_file(tmp_path: Path) -> None:
    """The output folder cannot be created under a file."""
    blocker = write(tmp_path / "blocker", "x")
    with pytest.raises(ValueError, match="cannot write"):
        write_calls(blocker / "out.json", [])


def test_output_keeps_unicode(tmp_path: Path) -> None:
    """Accents are written as is, the file is valid JSON."""
    output = tmp_path / "out.json"
    call = FunctionCall(prompt="Salue Zoé", name="fn_greet",
                        parameters={"name": "Zoé"})
    write_calls(output, [call])
    text = output.read_text(encoding="utf-8")
    assert "Zoé" in text
    assert json.loads(text)[0]["parameters"] == {"name": "Zoé"}


# ------------------------------------------------------------- CLI


def test_cli_reports_bad_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A broken input file: exit code 1, one clear line, no traceback."""
    bad = write(tmp_path / "bad.json", "[{]")
    assert cli.main(["--input", str(bad)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: invalid JSON")
    assert "Traceback" not in err


def test_cli_checks_output_before_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bad --output is reported before the model is even loaded."""
    def boom(cls: type, device: str | None = None) -> LanguageModel:
        """Fake LanguageModel.load that must never be called."""
        raise AssertionError("model loaded")

    monkeypatch.setattr(LanguageModel, "load", classmethod(boom))
    assert cli.main(["--output", str(tmp_path)]) == 1
    assert "cannot write" in capsys.readouterr().err


def test_cli_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ctrl-C while the model loads: exit code 130, clean message."""
    def interrupt(cls: type, device: str | None = None) -> LanguageModel:
        """Fake LanguageModel.load simulating Ctrl-C."""
        raise KeyboardInterrupt

    monkeypatch.setattr(LanguageModel, "load", classmethod(interrupt))
    assert cli.main(["--output", str(tmp_path / "o.json")]) == 130
    assert "Interrupted" in capsys.readouterr().err


def test_cli_unexpected_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Even an unforeseen bug ends with a message, not a traceback."""
    def broken(cls: type, device: str | None = None) -> LanguageModel:
        """Fake LanguageModel.load simulating an unforeseen bug."""
        raise RuntimeError("torch exploded")

    monkeypatch.setattr(LanguageModel, "load", classmethod(broken))
    assert cli.main(["--output", str(tmp_path / "o.json")]) == 1
    err = capsys.readouterr().err
    assert "Unexpected error (RuntimeError): torch exploded" in err


def test_cli_skips_a_failing_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One prompt that fails is reported; the others are still written."""
    from tests.test_generator import fake_loader

    def load(cls: type, device: str | None = None) -> LanguageModel:
        """Fake LanguageModel.load returning the tiny fake model."""
        return fake_loader(tmp_path)

    original = Generator.generate

    def flaky(self: Generator, request: str) -> FunctionCall:
        """Generator.generate that fails on 'Greet' prompts."""
        if request.startswith("Greet"):
            raise ValueError("no valid token left (dead end)")
        return original(self, request)

    monkeypatch.setattr(LanguageModel, "load", classmethod(load))
    monkeypatch.setattr(Generator, "generate", flaky)
    output = tmp_path / "out.json"
    assert cli.main(["--output", str(output)]) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(data) == 9  # 11 prompts, 2 "Greet ..." skipped
    assert capsys.readouterr().err.count("WARNING: skipped") == 2


def test_help_does_not_load_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--help exits before anything heavy happens."""
    def boom(cls: type, device: str | None = None) -> LanguageModel:
        """Fake LanguageModel.load that must never be called."""
        raise AssertionError("model loaded")

    monkeypatch.setattr(LanguageModel, "load", classmethod(boom))
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
