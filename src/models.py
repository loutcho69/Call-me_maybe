"""Pydantic models describing the input and output data of the program."""

from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Types JSON que notre decodeur contraint sait generer.
SUPPORTED_TYPES = ("number", "integer", "string", "boolean")
# Synonymes courants, ramenes aux types JSON.
TYPE_ALIASES = {
    "float": "number", "double": "number", "int": "integer",
    "bool": "boolean", "str": "string",
}

# Valeur Python d'un argument une fois genere.
# L'ordre compte : pydantic garde bool avant int, int avant float.
ArgValue = Union[bool, int, float, str]


class ParameterSpec(BaseModel):
    """Type description of one function parameter.

    Attributes:
        type: JSON type of the value, normalized (``"Float"`` -> ``number``).
        description: Optional human readable explanation.
    """

    model_config = ConfigDict(extra="ignore")

    type: str = Field(min_length=1)
    description: str | None = None

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, value: Any) -> Any:
        """Lowercase the type and map synonyms (int, float...) to JSON."""
        if isinstance(value, str):
            cleaned = value.strip().lower()
            return TYPE_ALIASES.get(cleaned, cleaned)
        return value

    @property
    def supported(self) -> bool:
        """True if the constrained decoder can generate this type."""
        return self.type in SUPPORTED_TYPES


class ReturnSpec(BaseModel):
    """Return type of a function: informative only, never generated.

    Any value is accepted (``array``, ``object``, ``null``...), so that an
    unusual return type never makes the whole file unusable.

    Attributes:
        type: The declared return type, as written in the file.
    """

    model_config = ConfigDict(extra="ignore")

    type: Any = None


class FunctionDefinition(BaseModel):
    """One callable function, as read from functions_definition.json.

    Attributes:
        name: Unique function name (e.g. ``fn_add_numbers``).
        description: What the function does; shown to the LLM.
        parameters: Mapping argument name -> type spec, in file order.
        returns: Declared return type (not used for generation).
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: ReturnSpec | None = None

    def unsupported_parameters(self) -> list[str]:
        """Return the parameters whose type cannot be generated."""
        return [
            name for name, spec in self.parameters.items()
            if not spec.supported
        ]


class PromptItem(BaseModel):
    """One test case from function_calling_tests.json.

    Attributes:
        prompt: Natural language request to translate.
    """

    model_config = ConfigDict(extra="ignore")

    prompt: str


class FunctionCall(BaseModel):
    """One entry of the output file.

    Attributes:
        prompt: The original natural language request.
        name: Name of the selected function.
        parameters: Generated arguments, typed per the definition.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, ArgValue]
