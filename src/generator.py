"""Constrained generation of one function call per prompt.

The JSON text is assembled piece by piece. Fixed parts (``{"name": "``,
``", "parameters": {"a": `` ...) are written by us; the model only
chooses the function name and each argument value, token by token,
among the tokens allowed by the matching constraint.
"""

import json

import numpy as np
from pydantic import BaseModel, Field

from src.constraints import (
    Constraint,
    NameConstraint,
    TokenIndex,
    make_value_constraint,
)
from src.llm import LanguageModel
from src.models import ArgValue, FunctionCall, FunctionDefinition
from src.prompt_builder import DEFAULT_STYLE, build_prompt


class GenerationStats(BaseModel):
    """Counters for the debug output and the performance analysis.

    Attributes:
        model_calls: Number of logits computations.
        forced_bytes: Bytes written without asking the model.
    """

    model_calls: int = 0
    forced_bytes: int = 0


class Generator(BaseModel):
    """Turns a natural language request into a FunctionCall.

    Attributes:
        model: The language model.
        index: Precomputed token facts for the masks.
        functions: Available functions.
        max_value_tokens: Safety limit of model calls per value.
        prompt_style: Prompt variant (see ``prompt_builder``).
        stats: Counters, cumulated over all prompts.
    """

    model: LanguageModel
    index: TokenIndex
    functions: list[FunctionDefinition] = Field(min_length=1)
    max_value_tokens: int = 80
    prompt_style: str = DEFAULT_STYLE
    stats: GenerationStats = Field(default_factory=GenerationStats)

    def generate(self, request: str) -> FunctionCall:
        """Produce the function call for one request.

        Args:
            request: The user's natural language request.

        Returns:
            The validated FunctionCall.

        Raises:
            ValueError: If the generation cannot finish.
        """
        prompt = build_prompt(self.functions, request, self.prompt_style)
        prefix_ids = self.model.encode(prompt)

        text = '{"name": "'
        names = [f.name.encode("utf-8") for f in self.functions]
        name = str(self._run(NameConstraint(names=names), prefix_ids, text))
        function = next(f for f in self.functions if f.name == name)
        text += name + '", "parameters": {'

        parameters: dict[str, ArgValue] = {}
        for position, (param, spec) in enumerate(function.parameters.items()):
            if position > 0:
                text += ", "
            # On s'arrete juste apres ":" : l'espace (et le guillemet
            # d'une chaine) font partie des tokens naturels du modele,
            # comme " -" ou ' "*'. La contrainte les accepte.
            text += json.dumps(param, ensure_ascii=False) + ":"
            constraint = make_value_constraint(spec.type)
            value = self._run(constraint, prefix_ids, text)
            parameters[param] = value
            text += " " + json.dumps(value, ensure_ascii=False)

        return FunctionCall(prompt=request, name=name, parameters=parameters)

    def _run(
        self, constraint: Constraint, prefix_ids: list[int], text: str
    ) -> ArgValue:
        """Let the model write one value under a constraint.

        Args:
            constraint: Fresh constraint for this value.
            prefix_ids: Encoded prompt.
            text: JSON written so far (the model sees it as context).

        Returns:
            The finished value.

        Raises:
            ValueError: If no token is allowed or the value never ends.
        """
        # On re-encode tout le texte JSON : le modele voit toujours un
        # decoupage naturel, et les tokens de fin (",  "}) ne sont jamais
        # gardes puisque c'est nous qui ecrivons la structure.
        ids = prefix_ids + self.model.encode(text)
        for _ in range(self.max_value_tokens):
            if constraint.done:
                break
            forced = constraint.forced_completion()
            if forced is not None:
                constraint.advance(forced)
                self.stats.forced_bytes += len(forced)
                continue
            token_id = self._pick_token(constraint, ids)
            constraint.advance(self.index.token_bytes[token_id])
            ids.append(token_id)
        if not constraint.done and not isinstance(constraint, NameConstraint):
            constraint.done = True  # limite atteinte : on coupe proprement
        if not constraint.done:
            raise ValueError("the function name was never completed")
        return constraint.value()

    def _pick_token(self, constraint: Constraint, ids: list[int]) -> int:
        """Greedy choice of the best allowed token.

        Args:
            constraint: Current constraint.
            ids: Token ids of the whole sequence so far.

        Returns:
            The chosen token id.

        Raises:
            ValueError: If the constraint allows no token at all.
        """
        mask = constraint.mask(self.index)
        if not mask.any():
            raise ValueError("no valid token left (dead end)")
        logits = self.model.logits(ids)
        self.stats.model_calls += 1
        if logits.shape[0] != mask.shape[0]:
            raise ValueError(
                f"model returned {logits.shape[0]} logits, "
                f"expected {mask.shape[0]}"
            )
        # Les tokens interdits recoivent -inf : argmax ne peut plus
        # les choisir, quelle que soit la preference du modele.
        masked = np.where(mask, logits, -np.inf)
        return int(np.argmax(masked))
