*This project has been created as part of the 42 curriculum by lobroue.*

# call me maybe

Function calling with a 0.6B-parameter language model, made reliable by
**constrained decoding**.

## Description

The program reads natural language requests such as *"What is the sum of
2 and 3?"* and a list of available functions, and outputs for each request
the function to call and its arguments, as JSON:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

It never computes the answer itself: it only translates the request into a
structured call.

The model is `Qwen/Qwen3-0.6B`, used through the provided `llm_sdk`. Such a
small model, asked to "answer in JSON", regularly produces broken JSON,
extra text or wrong types. Instead of hoping for a well-formed answer, the
program builds the answer token by token. At every step it **forbids every
token that would break the JSON or the schema** (its logit is set to
-infinity). The model keeps the choice among the valid tokens (which
function, which values), and the code guarantees the structure.

Results (details in [Performance analysis](#performance-analysis)):

- 100 % valid, schema-compliant JSON, even with a model that outputs random
  scores;
- 97.2 % exact answers on 36 test requests, 11/11 on the provided examples;
- about 1 to 3.5 seconds per request.

## Instructions

Requirements: Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
make install        # uv sync: creates .venv, installs numpy, pydantic, the SDK
make run            # uv run python -m src  (default input/output paths)
```

The first run downloads the model (about 1.5 GB) from Hugging Face.

Options (all optional):

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

The output folder is created if needed. Progress is printed on stderr.

| Rule | What it does |
|---|---|
| `make install` | install dependencies with uv |
| `make run` | run the program with the default files |
| `make debug` | run the program under `pdb` |
| `make clean` | remove `__pycache__`, `.mypy_cache`, `.pytest_cache` |
| `make lint` | `flake8 .` and `mypy .` with the required flags |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |
| `make test` | run the pytest suite (no model needed) |
| `make evaluate` | measure accuracy on test sets with known answers |
| `make check-vocab` | check our token decoding against the real tokenizer |

On a 42 computer with a small home quota, point the caches to `goinfre`
before `make install`: `export UV_CACHE_DIR=~/goinfre/uv HF_HOME=~/goinfre/hf`.

## Algorithm explanation

### 1. From token ids to bytes

Qwen uses byte-level BPE (like GPT-2). `vocab.json` (from
`get_path_to_vocab_file()`) maps token strings to ids. However, each byte is
stored as a printable character: a space is `Ġ`, a newline is `Ċ`, and
`é` is the two characters `Ã©`. `src/vocab.py` rebuilds GPT-2's
`bytes_to_unicode` table and inverts it, so that `token_bytes[id]` gives the
exact bytes each token writes (`"Ġsum"` → `b" sum"`). Work is done on
`bytes`, not `str`, because a token can hold half of a UTF-8 character.

Special tokens (`<|im_start|>`, `<think>`…, read from `tokenizer.json`)
and the ids beyond `vocab.json` are marked as unusable. The model returns
151,936 logits, but only 151,643 ids write text.

### 2. Precomputed token index

`TokenIndex` (`src/constraints.py`) scans the vocabulary once (≈ 0.15 s)
and stores categories as numpy arrays and lists:

- tokens that are safe anywhere inside a JSON string;
- the 54 tokens made only of digits, `.` and `-`;
- the 1,666 tokens containing a quote or a backslash;
- tokens that end a value (`,`, `}`, space…);
- a `bytes → id` lookup table.

Thanks to it, a mask over 151,936 tokens is built in 0.02 to 8 ms instead
of testing every token in Python.

### 3. One state machine per value

Each constraint answers `allows(bytes)`, `advance(bytes)` and
`mask(index)`:

| Constraint | Accepted | End |
|---|---|---|
| `NameConstraint` | text that stays a prefix of a known function name (trie-like lookup) | the name is complete and unambiguous, or a closing `"` |
| `NumberConstraint` | optional space, optional `-`, digits, at most one `.` (none for `integer`) | a token starting with `,` `}` or whitespace, once a digit exists |
| `StringConstraint` | optional space, opening `"`, then any text without raw `"`, `\` or control characters; only valid JSON escapes (`\"`, `\\`, `\n`…) | an unescaped `"`, followed only by separators |
| `BooleanConstraint` | prefixes of `true` / `false` | the word is complete |

A token is checked byte by byte against the state machine. One invalid
byte rejects the whole token, which is how multi-character tokens like
`",` or `"*` are handled.

Limits (40 characters for numbers, 200 bytes for strings, 80 tokens per
value) make sure every value terminates.

### 4. Generation loop

For each request (`src/generator.py`):

1. Build the chat prompt: system message with one line per function, the
   request, and an empty `<think></think>` block to disable Qwen3's slow
   reasoning mode.
2. Write `{"name": "` ourselves, then let the model write the name under
   `NameConstraint`. As soon as only one name is possible, the rest is
   written without calling the model.
3. For each parameter of the chosen function, in definition order, write
   `", "parameters": {"a":` ourselves, then let the model write the value
   under the constraint of its type.
4. At each model step:

   ```python
   logits = model.logits(ids)               # one score per token
   logits[~constraint.mask(index)] = -inf   # forbid invalid tokens
   token = argmax(logits)                   # greedy choice
   constraint.advance(token_bytes[token])
   ```

5. Build a pydantic `FunctionCall`. The output file is written with
   `json.dump`, never by string concatenation.

Before each value the whole JSON text written so far is re-encoded, so the
model always sees a natural tokenization. The token that ends a value (for
example `",`) is never kept: the structure is always written by the code.

## Design decisions

- **The model only writes values.** Every fixed part of the JSON is
  written by the code. This guarantees the structure and saves model calls
  (129 bytes forced on the 11 examples).
- **The function is chosen by the model**, among valid names only, never
  by keyword rules in Python.
- **Greedy decoding** (argmax). It is deterministic, reproducible and
  enough for extraction tasks.
- **Every class is a pydantic model**, as required: inputs, outputs,
  constraints, generator, statistics. `FunctionCall` forbids extra keys, so
  the output can only have `prompt`, `name` and `parameters`. A `number`
  is always written as a float (`2.0`).
- **Only public SDK methods**: `encode`, `decode` (checks only),
  `get_logits_from_input_ids`, `get_path_to_vocab_file`,
  `get_path_to_tokenizer_file`, and the `device` constructor parameter.
  The SDK is imported lazily, so `--help` and the tests do not load torch.
- **Tolerant function files**: the hidden test files may differ from the
  examples. Type synonyms (`float`, `int`, `bool`, `str`, any case) are
  mapped to JSON types. The return type is informative only, so any value
  (`array`, `object`, `null`...) is accepted. A function whose *parameter*
  type cannot be generated (`array`, `object`: nested arguments are a
  bonus feature) is skipped with a warning instead of rejecting the whole
  file.
- **No `\u` escapes in strings**: the model writes accented characters
  directly, and broken surrogate pairs become impossible.
- **The simplest prompt won.** Three prompt styles were measured (see
  below). The plain one was the most accurate and the fastest, so it is
  the default. The other two are kept in `src/prompt_builder.py` for
  comparison.
- **Errors never crash the program.** Input problems give one clear line
  and exit code 1, Ctrl-C gives exit code 130, and any unforeseen error is
  caught by a last safety net. The output path is checked *before*
  loading the model. A request whose generation fails is reported on
  stderr and skipped, so one bad request cannot cost the whole run. This
  never happened in any test, because the constraints never reach a dead
  end.

## Performance analysis

**Accuracy** (`make evaluate`): 36 requests with known answers, made of the
11 provided examples and 25 new ones. The new ones use different wordings,
negative and decimal numbers, names like `O'Brien`, uppercase vowels, and
4 extra functions with `integer` and `boolean` parameters.

| Prompt style | Examples | New requests | Total | Time (36 requests) |
|---|---|---|---|---|
| **basic** (default) | 11/11 | 24/25 | **97.2 %** | 48 s |
| rules | 10/11 | 24/25 | 94.4 % | 59 s |
| examples | 10/11 | 23/25 | 91.7 % | 80 s |

The remaining error: for *"Replace every digit in 'r2d2' with X"*, the
model uses `r2d2` as the regex. It was left as is, to avoid tuning the
prompt to our own test set.

**Speed** (Apple Silicon laptop):

| Device | Per request | 11 examples | Model calls per request |
|---|---|---|---|
| GPU (MPS, float16) | ≈ 1.3 s | 18 s including model loading | ≈ 11 |
| CPU only (float32) | ≈ 3.5 s | 36 s | ≈ 10 |

Accuracy is identical on CPU and GPU. Since the SDK has no cache, every
call re-reads the whole sequence (about 200 tokens on average), so the
cost is roughly *number of calls × sequence length*. Forcing the structure
and completing unambiguous names keeps the number of calls low.

**Reliability**: JSON validity does not depend on the model. Tests drive
the generator with a fake model returning random logits, over 20 random
seeds on a small vocabulary and 3 on the real Qwen vocabulary. Every
output is valid JSON with the exact parameter names and types.

## Challenges faced

- **Unnatural tokenization at value boundaries.** The first version wrote
  `"a": ` (with the space) and the opening quote of strings itself. Qwen
  has tokens like ` -`, ` "*` or ` true`, so the model was left in a
  position it never saw in training. `-8` became `8`, and `*` became ` *`.
  The fix: stop right after `:` and let the constraints accept an optional
  space and the opening quote. Accuracy went from **86.1 % to 97.2 %**,
  more than any prompt change.
- **Byte-level tokens.** Tokens are bytes, sometimes half a character.
  The vocabulary is decoded to bytes, and strings are kept as bytes until
  the end.
- **JSON escaping.** Prompts contain quotes (`"Hello 34 I'm 233 years
  old"`) and regexes contain backslashes (`\d+` must be written `"\\d+"`).
  The string state machine tracks the "after backslash" state and only
  allows valid escapes.
- **Ambiguous names.** If `fn_add` and `fn_add_numbers` both exist, after
  `fn_add` the model may close the quote or continue. Both paths are
  allowed and tested.
- **Words instead of symbols.** The small model wrote `asterisk` instead
  of `*`. Once the tokenization fix was in place, the plain prompt
  answered correctly. More rules or examples in the prompt made it worse.
- **Tooling.** Without a `.flake8` exclusion, `flake8 .` scanned `.venv`
  and crashed. The provided SDK is not typed and does not follow flake8,
  so it is excluded from both linters.

## Testing strategy

`make test` runs 94 pytest tests in a few seconds, without the model:

- **Vocabulary**: GPT-2 byte table, token decoding, special tokens.
- **Constraints**: hand-written tokens for precise cases (escaped quotes,
  `\d`, negative numbers, UTF-8, length limits, ambiguous names), plus
  random walks over the real vocabulary, where hundreds of random allowed
  tokens always give valid values. These run only if the Qwen tokenizer
  files are in the Hugging Face cache, and are skipped otherwise.
- **Generator and CLI**: a fake SDK model (byte tokenizer, random logits)
  checks that every output matches the schema for all types, a function
  without parameters, and a single function.
- **Robustness**: empty, broken, non UTF-8 or badly shaped files, unknown
  types, type synonyms, unusual return types, duplicate names, unwritable
  output, Ctrl-C, unexpected errors, a failing request, and `--help`.

With the real model:

- `make check-vocab` compares our decoding with `decode()` on 500 random
  tokens (491/491 identical; the others are partial UTF-8) and measures
  call times.
- `make evaluate` measures accuracy per prompt style, with
  `--device cpu` for a no-GPU check.

## Example usage

```bash
$ make run
uv run python -m src
Loading model...
[1/11]   1.1s fn_add_numbers {'a': 2.0, 'b': 3.0}
[2/11]   1.5s fn_add_numbers {'a': 265.0, 'b': 345.0}
[3/11]   0.7s fn_greet {'name': 'shrek'}
...
[9/11]   3.1s fn_substitute_string_with_regex {'source_string': "Hello 34 I'm 233 years old", 'regex': '([0-9]+)', 'replacement': 'NUMBERS'}
[10/11]  1.7s fn_substitute_string_with_regex {'source_string': 'Programming is fun', 'regex': '([aeiou])', 'replacement': '*'}
Model calls: 121, average sequence: 204 tokens, forced bytes: 129
Wrote 11/11 call(s) to data/output/function_calls.json in 18.3s
```

Custom files:

```bash
uv run python -m src --functions_definition my_functions.json \
    --input my_prompts.json --output results/calls.json
```

Error handling:

```bash
$ uv run python -m src --input missing.json
Error: file not found: missing.json

$ uv run python -m src --functions_definition bad.json
Error: invalid content in bad.json:
  - at 0.parameters.a.type: Input should be 'number', 'integer', 'string' or 'boolean'
```

Accuracy and speed without GPU:

```bash
uv run python -m scripts.evaluate --styles basic --device cpu
```

## Resources

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B): the
  model and its chat format.
- [Hugging Face LLM course, Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5).
- [GPT-2 `encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py):
  the original `bytes_to_unicode` table.
- [Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023)](https://arxiv.org/abs/2307.09702):
  constrained decoding with finite state machines over the vocabulary.
- [RFC 8259, The JSON Data Interchange Format](https://www.rfc-editor.org/rfc/rfc8259):
  string escapes and number grammar.
- [pydantic documentation](https://docs.pydantic.dev/latest/),
  [NumPy documentation](https://numpy.org/doc/stable/),
  [uv documentation](https://docs.astral.sh/uv/).

### How AI was used

An AI assistant (Claude, by Anthropic) was used as a pair-programming
partner throughout the project:

- **Planning**: splitting the project into phases, each with a checkable
  goal.
- **Learning**: explanations of byte-level BPE, logits and constrained
  decoding, and of the Python tools (pydantic, mypy, uv).
- **Code**: first drafts of the modules in `src/`, the tests in `tests/`
  and the scripts in `scripts/`.
- **Analysis**: reading the evaluation failures, which led to the
  tokenization fix described in *Challenges faced*, and drafting this
  README.

Every file was reviewed, run and tested on my machine. All model runs,
accuracy and timing figures above come from my own runs. The design
choices were discussed and validated before being kept, and I can explain
each part of the code.
