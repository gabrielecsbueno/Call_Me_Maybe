from pydantic import BaseModel, model_validator
from enum import Enum


class TypeModel(Enum):
    """
    Tipos de dato admitidos para parámetros y retornos de funciones.

    Attributes
    ----------
    number : str
        Representa un valor numérico (int o float).
    string : str
        Representa una cadena de caracteres.
    boolean : str
        Representa un valor lógico (``true``/``false``).
    """
    number = "number"
    string = "string"
    boolean = "boolean"
    integer = "integer"


class ParamDef(BaseModel):
    """
    Definición de un parámetro de función, validada por Pydantic.

    Attributes
    ----------
    type : TypeModel
        Tipo de dato esperado para el parámetro.
    """
    type: TypeModel


class FunctionDef(BaseModel):
    """
    Definición completa de una función disponible para la LLM, validada
    por Pydantic.

    Attributes
    ----------
    name : str
        Nombre de la función (ej.: ``"fn_add_numbers"``).
    description : str
        Descripción en lenguaje natural de lo que hace la función.
    parameters : dict[str, ParamDef]
        Relaciona el nombre de cada parámetro con su definición de tipo.
    returns : dict[str, TypeModel]
        Tipo del valor que devuelve la función.
    """
    name: str
    description: str
    parameters: dict[str, ParamDef]
    returns: dict[str, TypeModel]

    @model_validator(mode='after')
    def validator_function(self) -> 'FunctionDef':
        """
        Comprueba si los valores recibidos en las funciones están vacíos.
        """
        if self.name == "":
            raise ValueError("el 'name' está vacío.")
        if self.description == "":
            raise ValueError("el 'description' está vacío.")
        if not self.returns:
            raise ValueError("el 'returns' está vacío.")
        if not self.parameters:
            raise ValueError("el 'parameters' está vacío.")
        return self


class FunctionCall(BaseModel):
    """
    Una llamada/pregunta en lenguaje natural hecha por el usuario,
    validada por Pydantic.

    Attributes
    ----------
    prompt : str
        Texto de la pregunta del usuario.
    """
    prompt: str

    @model_validator(mode='after')
    def validator_prompt(self) -> 'FunctionCall':
        """
        Comprueba si el valor del 'prompt' está vacío
        """
        if self.prompt == "":
            raise ValueError("el 'prompt' está vacío.")
        return self
