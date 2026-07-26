from .models import FunctionDef, FunctionCall
import json
import os


def load_functions(file_path: str) -> list[FunctionDef]:
    """
    Carga las definiciones de funciones del archivo JSON.

    Parameters
    ----------
    file_path : string
        Ruta al archivo JSON que contiene las funciones válidas para su
        uso.

    Returns
    -------
    list[FunctionDef]
        Lista de objetos FunctionDef, de funciones validadas por
        Pydantic.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe o el JSON es inválido.
    """
    try:

        with open(file_path, "r") as file:
            data = json.load(file)

        if not data or not isinstance(data, list):
            raise ValueError(f"el archivo '{file_path}' debería contener "
                             "una lista de objetos")

    except ValueError:
        raise ValueError(f"el archivo '{file_path}' tiene un formato JSON "
                         "inválido.")

    except FileNotFoundError:
        raise FileNotFoundError(f"no se encontró el archivo '{file_path}'.")

    except json.JSONDecodeError as err:
        raise FileNotFoundError(f"el archivo '{file_path}' es un JSON "
                                f"inválido ({err})")

    except PermissionError:
        raise PermissionError(f"el archivo '{file_path}' está bloqueado "
                              "para lectura.")

    except Exception as err:
        raise Exception(f"Ocurrió un error inesperado: {err}")

    function_list = []

    for item in data:
        function_list.append(
            FunctionDef(
                name=item.get('name'),
                description=item.get('description'),
                parameters=item.get('parameters'),
                returns=item.get('returns')
            )
        )

    return function_list


def load_calls(file_path: str) -> list[FunctionCall]:
    """
    Carga las definiciones de las llamadas del archivo JSON.

    Parameters
    ----------
    file_path : str
        Ruta al archivo JSON de llamadas/prompts (ej.:
        function_calling_tests.json), recibida del usuario vía --input.

    Returns
    -------
    list[FunctionCall]
        Lista de objetos FunctionCall validados por Pydantic.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe o el JSON es inválido.
    ValueError
        Si el archivo no contiene una lista de objetos, o si algún item
        no tiene el campo 'prompt'.
    PermissionError
        Si el archivo está bloqueado para lectura.
    """
    try:

        with open(file_path, "r") as file:
            data: list[dict[str, str]] = json.load(file)

        if not data or not isinstance(data, list):
            raise ValueError(f"el archivo '{file_path}' debería contener "
                             "una lista de objetos")

    except ValueError:
        raise ValueError(f"el archivo '{file_path}' tiene un formato JSON "
                         "inválido.")

    except FileNotFoundError:
        raise FileNotFoundError(f"no se encontró el archivo '{file_path}'.")

    except json.JSONDecodeError as err:
        raise FileNotFoundError(f"el archivo '{file_path}' es un JSON "
                                f"inválido ({err})")

    except PermissionError:
        raise PermissionError(f"el archivo '{file_path}' está bloqueado "
                              "para lectura.")

    except Exception as err:
        raise Exception(f"Ocurrió un error inesperado: {err}")

    else:
        call_list = []

        for item in data:
            prompt_value = item.get('prompt')

            if prompt_value is None:
                raise ValueError(f"el archivo '{file_path}' tiene un item "
                                 "sin el campo 'prompt'.")

            call_list.append(
                FunctionCall(
                    prompt=prompt_value
                )
            )

    return call_list


def make_json(json_texts: list[str], file_path: str) -> None:
    """
    Combina las respuestas generadas por la LLM (una cadena JSON por
    prompt) en una única lista y las guarda en `file_path`.

    Ejemplo del archivo generado::

        [
            {
                "prompt": "What is the sum of 2 and 3?",
                "name": "fn_add_numbers",
                "parameters": {"a": 2.0, "b": 3.0}
            }
        ]

    Parameters
    ----------
    json_texts : list[str]
        Lista de cadenas, cada una un JSON válido generado por la LLM
        para un prompt.
    file_path : str
        Ruta del archivo de salida a crear/sobrescribir.

    Returns
    -------
    None

    Raises
    ------
    PermissionError
        Si el archivo/directorio de destino está bloqueado para
        escritura.
    FileNotFoundError
        Si el directorio de destino no existe.
    IsADirectoryError
        Si `file_path` apunta a un directorio en vez de un archivo.
    """
    data = []

    try:
        for item in json_texts:
            data.append(json.loads(item))

        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    except PermissionError:
        raise PermissionError("permiso denegado para guardar el archivo "
                              f"'{file_path}'.")

    except FileNotFoundError:
        raise FileNotFoundError("el directorio de destino no existe.")

    except IsADirectoryError:
        raise IsADirectoryError("la ruta indicada es un directorio, no "
                                "un nombre de archivo válido.")

    except json.JSONDecodeError:
        raise FileNotFoundError(f"el '{item}' generado por la llm es "
                                "inválido")

    except Exception as err:
        raise Exception(f"Ocurrió un error inesperado: {err}")
