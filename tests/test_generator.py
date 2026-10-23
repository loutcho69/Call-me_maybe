"""End-to-end tests of the generator and the CLI with a random model."""

import json
from pathlib import Path

import pytest

from src import cli
from src.constraints import TokenIndex
from src.generator import Generator
from src.io_utils import load_functions, load_prompts
from src.llm import LanguageModel
from src.models import FunctionCall, FunctionDefinition
from tests.conftest import find_tokenizer_dir
from tests.fakes import FakeBackend, tiny_vocabulary

ROOT = Path(__file__).resolve().parent.parent
FUNCTIONS = load_functions(ROOT / "data/input/functions_definition.json")
PROMPTS = [p.prompt for p in
           load_prompts(ROOT / "data/input/function_calling_tests.json")]
PYTHON_TYPES = {"number": float, "integer": int, "string": str,
                "boolean": bool}

EXTRA_FUNCTIONS = [
    FunctionDefinition.model_validate({
        "name": "fn_set_flag",
        "description": "Set a flag and a counter.",
        "parameters": {"on": {"type": "boolean"},
                       "count": {"type": "integer"}},
    }),
    FunctionDefinition.model_validate(
        {"name": "fn_ping", "description": "Ping, no argument."}
    ),
]


def check_call(
    call: FunctionCall, functions: list[FunctionDefinition]
) -> None:
    """Assert that a call matches its function's schema exactly."""
    function = next(f for f in functions if f.name == call.name)
    assert list(call.parameters) == list(function.parameters)
    for name, spec in function.parameters.items():
        assert type(call.parameters[name]) is PYTHON_TYPES[spec.type]
    dumped = call.model_dump()
    assert list(dumped) == ["prompt", "name", "parameters"]
    assert json.loads(json.dumps(dumped)) == dumped


def tiny_generator(
    seed: int, functions: list[FunctionDefinition]
) -> Generator:
    """Generator with the tiny vocabulary and random logits."""
    vocab, specials = tiny_vocabulary()
    backend = FakeBackend(vocab=vocab, specials=specials, seed=seed)
    return Generator(model=LanguageModel(backend=backend),
                     index=TokenIndex.build(vocab), functions=functions)


@pytest.mark.parametrize("seed", range(20))
def test_random_model_always_valid(seed: int) -> None:
    """Whatever the model prefers, every call fits the schema."""
    generator = tiny_generator(seed, FUNCTIONS)
    for prompt in PROMPTS:
        check_call(generator.generate(prompt), FUNCTIONS)


@pytest.mark.parametrize("seed", range(10))
def test_integer_boolean_and_no_parameter(seed: int) -> None:
    """Types not in the example files work too."""
    functions = FUNCTIONS + EXTRA_FUNCTIONS
    generator = tiny_generator(seed, functions)
    for prompt in ("turn the flag on 3 times", "ping"):
        check_call(generator.generate(prompt), functions)


def test_single_function_needs_no_model_call_for_name() -> None:
    """With one function, the name is written directly."""
    generator = tiny_generator(0, [EXTRA_FUNCTIONS[1]])
    call = generator.generate("ping")
    assert call.name == "fn_ping" and call.parameters == {}
    assert generator.stats.model_calls == 0


def test_real_vocabulary_random_model(tmp_path: Path) -> None:
    """Same check with the real Qwen vocabulary (skipped if absent)."""
    folder = find_tokenizer_dir()
    if folder is None:
        pytest.skip("Qwen tokenizer files not found")
    from src.vocab import load_vocabulary
    from tests.conftest import QWEN_LOGITS_SIZE

    vocab = load_vocabulary(folder / "vocab.json",
                            folder / "tokenizer.json", QWEN_LOGITS_SIZE)
    specials = {"<|im_start|>": 151644, "<|im_end|>": 151645,
                "<think>": 151667, "</think>": 151668}
    index = TokenIndex.build(vocab)
    for seed in range(3):
        backend = FakeBackend(vocab=vocab, specials=specials, seed=seed)
        generator = Generator(model=LanguageModel(backend=backend),
                              index=index, functions=FUNCTIONS)
        for prompt in PROMPTS:
            check_call(generator.generate(prompt), FUNCTIONS)


# ------------------------------------------------------------------ CLI


def fake_loader(tmp_path: Path) -> LanguageModel:
    """A LanguageModel.load replacement using the tiny fake."""
    vocab, specials = tiny_vocabulary()
    backend = FakeBackend(vocab=vocab, specials=specials,
                          folder=tmp_path / "tok")
    return LanguageModel(backend=backend)


def test_cli_end_to_end(tmp_path: Path,
                        monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI writes one valid entry per prompt, output folder created."""
    monkeypatch.setattr(LanguageModel, "load",
                        classmethod(lambda cls: fake_loader(tmp_path)))
    output = tmp_path / "new_dir" / "calls.json"
    assert cli.main(["--output", str(output)]) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert [entry["prompt"] for entry in data] == PROMPTS
    for entry in data:
        check_call(FunctionCall.model_validate(entry), FUNCTIONS)


def test_cli_no_prompt_skips_model(tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero prompt: empty output, the model is never loaded."""
    def boom(cls: type) -> LanguageModel:
        raise AssertionError("model should not be loaded")

    monkeypatch.setattr(LanguageModel, "load", classmethod(boom))
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    output = tmp_path / "out.json"
    assert cli.main(["--input", str(empty), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == []


def test_cli_model_load_failure(tmp_path: Path,
                                monkeypatch: pytest.MonkeyPatch,
                                capsys: pytest.CaptureFixture[str]) -> None:
    """A model that cannot load gives a clean error and exit code 1."""
    def fail(cls: type) -> LanguageModel:
        raise ValueError("cannot load model 'x': no network")

    monkeypatch.setattr(LanguageModel, "load", classmethod(fail))
    assert cli.main(["--output", str(tmp_path / "o.json")]) == 1
    assert "no network" in capsys.readouterr().err
