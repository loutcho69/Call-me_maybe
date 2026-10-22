"""Thin wrapper around the provided ``llm_sdk`` (public methods only)."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


class LanguageModel(BaseModel):
    """Access to the LLM through the four operations we need.

    The SDK object is kept as ``Any`` because the SDK is not typed.
    Only its public methods are used: ``encode``, ``decode``,
    ``get_logits_from_input_ids`` and the ``get_path_to_*`` helpers.

    Attributes:
        backend: The ``Small_LLM_Model`` instance (or a fake in tests).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: Any

    @classmethod
    def load(cls, model_name: str = DEFAULT_MODEL) -> "LanguageModel":
        """Load the real model through the SDK.

        The SDK is imported here and not at the top of the file: importing
        it loads torch, which takes seconds. ``--help`` and the tests stay
        fast this way.

        Args:
            model_name: Hugging Face identifier of the model.

        Returns:
            A ready to use LanguageModel.

        Raises:
            ValueError: If the SDK or the model cannot be loaded.
        """
        try:
            from llm_sdk import Small_LLM_Model

            backend = Small_LLM_Model(model_name)
        except Exception as err:  # the SDK can fail in many ways
            raise ValueError(
                f"cannot load model {model_name!r}: {err}"
            ) from None
        return cls(backend=backend)

    def encode(self, text: str) -> list[int]:
        """Turn text into token ids.

        Args:
            text: Text to tokenize.

        Returns:
            The list of token ids.
        """
        # Le SDK renvoie un tenseur 2D [[ids...]] : .tolist()[0]
        # donne une liste Python, sans jamais importer torch nous-memes.
        rows = self.backend.encode(text).tolist()
        return [int(token_id) for token_id in rows[0]]

    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into text (used for checks only).

        Args:
            ids: Token ids.

        Returns:
            The decoded text.
        """
        return str(self.backend.decode(ids))

    def logits(self, ids: list[int]) -> npt.NDArray[np.float32]:
        """Score of every vocabulary token for the position after ``ids``.

        Args:
            ids: The whole sequence so far (the SDK has no cache).

        Returns:
            A 1D float array, one score per token id.
        """
        raw = self.backend.get_logits_from_input_ids(ids)
        return np.asarray(raw, dtype=np.float32)

    def vocab_path(self) -> Path:
        """Return the path of the ``vocab.json`` file."""
        return Path(self.backend.get_path_to_vocab_file())

    def tokenizer_path(self) -> Path:
        """Return the path of the ``tokenizer.json`` file."""
        return Path(self.backend.get_path_to_tokenizer_file())
