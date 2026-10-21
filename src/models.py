"""Pydantic models describing the input and output data of the program."""

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# Types JSON que notre decodeur contraint saura generer.
ParamType = Literal["number", "integer", "string", "boolean"]

# Valeur Python d'un argument une fois genere.
# L'ordre compte : pydantic garde bool avant int, int avant float.
ArgValue = Union[bool, int, float, str]


class ParameterSpec(BaseModel):
    """Type description of one function parameter (or return value).

    Attributes:
        type: JSON type of the value.
        description: Optional human readable explanation.
    """

    model_config = ConfigDict(extra="ignore")

    type: ParamType
    description: str | None = None


class FunctionDefinition(BaseModel):
    """One callable function, as read from functions_definition.json.

    Attributes:
        name: Unique function name (e.g. ``fn_add_numbers``).
        description: What the function does; shown to the LLM.
        parameters: Mapping argument name -> type spec, in file order.
        returns: Type of the returned value.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: ParameterSpec | None = None


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
