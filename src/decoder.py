from llm_sdk import Small_LLM_Model  # type: ignore
from .models import FunctionDef, FunctionCall, ParamDef
import numpy
import json
from typing import Callable


def get_vocab(model: Small_LLM_Model) -> tuple[dict[str, int], dict[int, str]]:
    """
    Carga el vocabulario del archivo devuelto por
    ``model.get_path_to_vocab_file()`` y arma los dos mapeos entre token
    e id.

    Parameters
    ----------
    model : Small_LLM_Model
        Instancia del modelo, usada para localizar el archivo de
        vocabulario.

    Returns
    -------
    tuple[dict[str, int], dict[int, str]]
        Un diccionario ``{texto_del_token: id}`` y su inverso
        ``{id: texto_del_token}``.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe o el JSON es inválido.
    PermissionError
        Si el archivo está bloqueado para lectura.
    """

    try:
        # 1. busco la ruta del archivo de vocabulario
        vocab_path = model.get_path_to_vocab_file()

        # 2. abro el archivo en modo lectura
        with open(vocab_path, "r") as file:
            vocab_token_to_id = json.load(file)

            # .load me devuelve un dict: 'token_string': 'id'

        # 3. invierto para -> id: token
        vocab_id_to_token = {i: t for t, i in vocab_token_to_id.items()}

    except FileNotFoundError:
        raise FileNotFoundError(f"no se encontró el archivo '{vocab_path}'.")

    except json.JSONDecodeError as err:
        raise FileNotFoundError(f"el archivo '{vocab_path}' es un JSON "
                                f"inválido ({err})")

    except PermissionError:
        raise PermissionError(f"el archivo '{vocab_path}' está bloqueado "
                              "para lectura.")

    except Exception as err:
        raise Exception(f"Ocurrió un error inesperado: {err}")

    return vocab_token_to_id, vocab_id_to_token


def valid_tokens_for_function_name(
        vocab: dict[str, int],
        function_defs: list[FunctionDef],
        current_prefix: str
        ) -> set[int]:
    """
    Filtra los tokens del vocabulario que pueden continuar un nombre de
    función válido, dado lo que ya se generó hasta ahora.

    Un token ``t`` es válido si ``current_prefix + t`` es prefijo de al
    menos un nombre de función válido — no una comparación exacta con el
    nombre completo. Ejemplo, con ``valid_names = ["fn_add_numbers",
    "fn_greet"]`` y ``current_prefix = "fn_"``: el token ``"add"`` se
    acepta porque ``"fn_add"`` es prefijo de ``"fn_add_numbers"``; el
    token ``"hello"`` se rechaza porque ningún nombre empieza con
    ``"fn_hello"``.

    Parameters
    ----------
    vocab : dict[str, int]
        Diccionario con tokens e ids del vocabulario del modelo.
    function_defs : list[FunctionDef]
        Lista de definiciones de funciones válidas (del archivo JSON).
    current_prefix : str
        Texto ya generado por la LLM hasta el momento.

    Returns
    -------
    set[int]
        Conjunto de ids de los tokens válidos.
    """

    valid_function_names = [func.name for func in function_defs]

    valid_token_ids = set(i for t, i in vocab.items()
                          if any(name.startswith(current_prefix + t)
                          for name in valid_function_names))

    return valid_token_ids


def valid_tokens_for_number(
      vocab: dict[str, int],
      current_prefix: str,
      delimiter: str) -> set[int]:
    """
    Filtra los tokens del vocabulario válidos para componer un valor
    numérico.

    Un token es válido si todos sus caracteres son dígitos, punto o
    signo negativo (sin repetir el punto decimal ni poner el signo en
    medio del número), o si el token es exactamente el delimitador — que
    solo sirve de señal de parada y nunca se escribe en el valor
    generado.

    Parameters
    ----------
    vocab : dict[str, int]
        Diccionario con tokens e ids del vocabulario del modelo.
    current_prefix : str
        Texto ya generado por la LLM hasta el momento.
    delimiter : str
        Token que señala "terminé el número" (``','`` si hay más
        parámetros, ``'}'`` si es el último).

    Returns
    -------
    set[int]
        Conjunto de ids de los tokens válidos.
    """
    # has_dot = '.' in current_prefix

    valid_token_ids = set(i for t, i in vocab.items()
                          if (all(c in '0123456789.-' for c in t)
                              and not (current_prefix and '-' in t))
                          or t == delimiter)
    return valid_token_ids


def valid_tokens_for_string(
        vocab: dict[str, int],
        current_prefix: str
) -> set[int]:
    """
    Filtra los tokens del vocabulario válidos para componer el
    contenido de una cadena.

    Son válidos todos los tokens excepto los que contengan comillas sin
    escapar, barra invertida o caracteres de control (simplificación
    válida). La comilla aislada también se acepta, ya que es la señal
    de cierre que ``should_stop`` espera recibir.

    Esta comprobación mira la clave cruda de `vocab` (lo que guarda el
    `vocab.json` del tokenizador), no el texto real tras la
    decodificación — los tokenizadores byte-level BPE remapean los
    bytes de control a otros caracteres dentro del `vocab.json`, así que
    por sí sola no cubre todos los casos. Sirve de primer filtro,
    barato, aplicado antes del argmax; `invalid_token_for_string`
    complementa después, mirando el texto ya decodificado del token
    elegido.

    Parameters
    ----------
    vocab : dict[str, int]
        Diccionario con tokens e ids del vocabulario del modelo.
    current_prefix : str
        Texto ya generado por la LLM hasta el momento.

    Returns
    -------
    set[int]
        Conjunto de ids de los tokens válidos.
    """
    valid_token_ids = set(i for t, i in vocab.items()
                          if t == '"' or (
                               '"' not in t
                               and '\\' not in t
                               and all(
                                    ord(c) >= 0x20 for c in t
                               )
                               and not (
                                   current_prefix == "" and t[:1].isspace()
                                )
                          )
                          )

    return valid_token_ids


def invalid_token_for_string(text_id: str) -> bool:
    """
    Comprueba si el texto real de un token (ya decodificado) no debería
    aceptarse dentro del contenido de una cadena, aunque haya pasado el
    filtro de `valid_tokens_for_string`.

    Necesario porque los tokenizadores byte-level BPE (como el de Qwen)
    remapean los bytes de control a otros caracteres dentro del
    `vocab.json` — un token puede parecer seguro en la clave cruda y aun
    así decodificar a un carácter de control real (salto de línea,
    tabulación, etc.), que solo aparece de verdad tras
    `model.decode(...)`.

    Parameters
    ----------
    text_id : str
        Texto decodificado del token elegido por el argmax.

    Returns
    -------
    bool
        True si el token debe descartarse (contiene comilla, barra
        invertida o carácter de control fuera de la señal de cierre
        legítima).
    """
    if text_id == '"':
        return False

    return ('"' in text_id or '\\' in text_id
            or any(ord(c) < 0x20 for c in text_id))


def valid_tokens_for_boolean(
        vocab: dict[str, int]
) -> set[int]:
    """
    Filtra los tokens del vocabulario que pueden continuar un valor
    booleano válido (``"true"`` o ``"false"``), dado lo que ya se
    generó hasta ahora.

    Misma lógica de coincidencia por prefijo de
    `valid_tokens_for_function_name`, aplicada al conjunto finito
    ``("true", "false")``.

    Parameters
    ----------
    vocab : dict[str, int]
        Diccionario con tokens e ids del vocabulario del modelo.
    current_prefix : str
        Texto ya generado por la LLM hasta el momento.

    Returns
    -------
    set[int]
        Conjunto de ids de los tokens válidos.
    """
    valid_values = ("true", "false")
    valid_token_ids = set(i for t, i in vocab.items()
                          if t in valid_values
                          )
    return valid_token_ids


def make_prompt(function_defs: list[FunctionDef],
                function_calls: list[FunctionCall]
                ) -> list[str]:
    """
    Construye los prompts enviados a la LLM, uno por cada llamada del
    usuario.

    Formato del prompt::

        You have the following functions available:
        <lista de funciones (nombre(parámetros): Descripción)>
        Given the user request: <pregunta>
        The answer must be in the following format:
        {"name": "<function_name>", "parameters": {...}}

    Parameters
    ----------
    function_defs : list[FunctionDef]
        Lista de definiciones de funciones válidas (del archivo JSON).
    function_calls : list[FunctionCall]
        Lista de llamadas/preguntas del usuario (del archivo JSON).

    Returns
    -------
    list[str]
        Lista de prompts, uno por cada item de `function_calls`.
    """
    functions_available: str = ""

    for item in function_defs:
        params_str = ', '.join(f'{k}: {v.type.value}'
                               for k, v in item.parameters.items())

        functions_available += (f'- {item.name} ({params_str}): '
                                f'{item.description}\n')

    prompts: list[str] = []

    for call in function_calls:
        text_prompt = ("You have the following functions available:\n"
                       f"{functions_available}Given the user request: "
                       f"{call.prompt}\nThe answer must be in the following "
                       'format: {"name": "<function_name>", "parameters": '
                       "{...}}")

        prompts.append(text_prompt)

    return prompts


def encode_prompt(model: Small_LLM_Model, text: str) -> list[int]:
    """
    Tokeniza un texto y devuelve la lista de ids correspondiente.

    Parameters
    ----------
    model : Small_LLM_Model
        Instancia del modelo, usada para tokenizar el texto.
    text : str
        Texto a tokenizar.

    Returns
    -------
    list[int]
        Lista de ids numéricos de los tokens del texto.
    """

    # Texto -> Tokens
    input_tensor = model.encode(text)

    # encode devuelve una lista de ids numéricos como un tensor
    # con formato [[id1, id2, id3...]]

    # convertimos el tensor a lista de Python y tomamos la primera fila [0]
    input_ids_list: list[int] = input_tensor[0].tolist()

    return input_ids_list


def restrict_logits(
        logits: list[float],
        valid_token_ids: set[int]
        ) -> list[float]:
    """
    Aplica la máscara de decodificación restringida sobre los logits,
    poniendo ``-inf`` en todo id que no esté en `valid_token_ids`.

    Parameters
    ----------
    logits : list[float]
        Logits originales del modelo para el próximo token, indexados
        por id de vocabulario.
    valid_token_ids : set[int]
        Conjunto de ids de tokens permitidos en esta posición.

    Returns
    -------
    list[float]
        Logits con ``-inf`` en toda posición cuyo id no esté en
        `valid_token_ids`.
    """
    return [x if vocab_idx in valid_token_ids else float('-inf')
            for vocab_idx, (x) in enumerate(logits)]


def generate_tokens(model: Small_LLM_Model,
                    idx: list,
                    should_stop: Callable,
                    get_valid_tokens: Callable,
                    is_token_invalid: Callable[[str], bool] | None = None
                    ) -> str:
    """
    Genera tokens uno a uno, aplicando decodificación restringida, hasta
    que `should_stop` señale el final.

    En cada iteración: recalcula los ids válidos (vía
    `get_valid_tokens`), obtiene los logits del próximo token, enmascara
    los inválidos, elige el token de mayor logit, lo decodifica a texto
    y decide si para o sigue acumulando.

    Parameters
    ----------
    model : Small_LLM_Model
        Instancia del modelo, usada para obtener logits y decodificar
        tokens.
    idx : list
        Secuencia de ids de entrada, mutada in-place con cada token
        aceptado.
    should_stop : Callable[[str, str], bool]
        Recibe el texto ya acumulado (``prompt``) y el token recién
        decodificado (``text_id``); devuelve ``True`` cuando la
        generación debe parar.

        - nombre de función: el criterio es sobre el `prompt` ya
          acumulado (tiene que ser ya un nombre válido completo — el
          ``text_id`` de esta llamada es solo el "espía" que confirma
          que terminó, y se descarta, porque el nombre completo ya está
          en `prompt`).
        - número/cadena/booleano: el criterio es sobre el token aislado
          (el modelo "señala" que terminó emitiendo el delimitador
          esperado). Ese token de señalización se descarta — no entra
          en `prompt` ni en `idx`, quien inyecta el delimitador de
          verdad es el código externo, para no duplicarlo.
    get_valid_tokens : Callable[[str], set[int]]
        Recibe el `prompt` acumulado y devuelve el conjunto de ids
        válidos para el próximo token, recalculado en cada iteración.
    is_token_invalid : Callable[[str], bool], optional
        Recibe el texto ya decodificado del token elegido por el argmax
        y devuelve ``True`` si no debe aceptarse. Sirve para detectar
        casos que `get_valid_tokens` no puede anticipar (ej.: un
        carácter de control que solo aparece después de la
        decodificación real del tokenizador). Cuando devuelve ``True``,
        ese id se descarta (pasa a ``-inf``) y se elige un nuevo argmax,
        sin gastar una nueva llamada al modelo. Si es ``None``, ningún
        token se descarta de esta forma.

    Returns
    -------
    str
        Texto acumulado hasta la señal de parada (sin el token de
        señalización).
    """
    prompt = ""
    while True:

        # 1. define el valor de los ids válidos
        valid_ids = get_valid_tokens(prompt)

        # 2. obtiene los logits del modelo y los logits válidos
        original_logits = model.get_logits_from_input_ids(idx)

        # 3. pone -infinito en los logits de los tokens inválidos
        filtered_logits = restrict_logits(original_logits, valid_ids)

        # 4. elige el token con mayor logit restante, descartando y
        # probando de nuevo si el texto decodificado es inválido (límite
        # de intentos = tamaño del vector, para nunca trabarse aquí)
        for _ in range(len(filtered_logits)):
            max_id = int(numpy.argmax(filtered_logits))

            # 4. decodificamos el token a texto, decode espera
            # torch.Tensor o list[int]
            text_id = model.decode([max_id])

            if is_token_invalid is None or not is_token_invalid(text_id):
                break

            filtered_logits[max_id] = float('-inf')

        # 6. comprueba si este token es la señal de parada antes de
        # confirmarlo en el prompt/ids
        if should_stop(prompt, text_id):
            break

        # 7. lo agregamos a nuestro texto
        prompt += text_id

        # 8. lo agregamos a la secuencia de ids
        idx.append(max_id)

    return prompt


def constrained_decoding(
        model: Small_LLM_Model,
        vocab: dict[str, int],
        input_ids_list: list,
        function_defs: list[FunctionDef],
        call_prompt: str
        ) -> str:
    """
    Genera la respuesta de la LLM en JSON válido, campo a campo, usando
    decodificación restringida (IDs -> LLM -> Logits -> elección del
    próximo token).

    Parameters
    ----------
    model : Small_LLM_Model
        Instancia del modelo usada para generar los tokens.
    vocab : dict[str, int]
        Diccionario con tokens e ids del vocabulario del modelo.
    input_ids_list : list
        Ids de entrada (prompt inicial ya tokenizado); mutado in-place a
        medida que avanza la generación.
    function_defs : list[FunctionDef]
        Lista de definiciones de funciones válidas, usada para
        restringir el nombre generado y descubrir los parámetros
        esperados.
    call_prompt : str
        Pregunta original del usuario, inyectada en el campo
        ``"prompt"`` del JSON de salida (escapada con `json.dumps`).

    Returns
    -------
    str
        Texto generado (JSON), decodificado solo a partir de la
        posición en la que empezó `input_ids_list` (sin repetir el
        prompt de entrada).

    Raises
    ------
    ValueError
        Si el nombre generado por la LLM no corresponde a ninguna
        función en `function_defs`.
    """

    # 1. guardo el tamaño del prompt para quitarlo al final
    initial_input_size = len(input_ids_list)

    # 2. inyecta '{"prompt": <prompt>, "name": "'
    call_prompt_json = json.dumps(call_prompt)
    input_ids_list.extend(encode_prompt(model, '{"prompt": '))
    input_ids_list.extend(
        encode_prompt(model, f'{call_prompt_json}, "name": "'))

    # 3. genera el nombre de la función con restricciones - para cuando
    # el nombre está completo
    function_names = [func.name for func in function_defs]
    generated_name = generate_tokens(
        model,
        input_ids_list,
        lambda prompt, text_id: prompt in function_names,
        lambda prompt: valid_tokens_for_function_name(
            vocab, function_defs, prompt)
        )

    # 4. busco qué función eligió
    chosen_function = next(
        (func for func in function_defs if func.name == generated_name),
        None)

    # 5. trato un error en caso de que no devuelva nada
    if chosen_function is None:
        raise ValueError(f"el nombre generado '{generated_name}' no "
                         "corresponde a ninguna función definida")

    # hasta aquí - {"prompt": <prompt>, "name": "<generated_name>

    # 6. inyecta '", "parameters": {' en los ids
    input_ids_list.extend(encode_prompt(model, '", "parameters": {'))

    # hasta aquí - {"prompt": <prompt>, "name": "<generated_name>",
    # "parameters": {

    # 5. según la función elegida busco qué parámetros son necesarios
    params: list[tuple[str, ParamDef]] = list(
        chosen_function.parameters.items())

    for i, (param_name, param_info) in enumerate(params):

        # a. inyecta '"param_name": ' en los ids
        input_ids_list.extend(encode_prompt(model, f'"{param_name}": '))

        # hasta aquí - {"prompt": <prompt>, "name": "<generated_name>",
        # "parameters": { <param_name>:

        param_type = param_info.type.value

        if param_type == "number" or param_type == "integer":

            delimiter = '}' if i == len(params) - 1 else ','

            generated_number = generate_tokens(
                model,
                input_ids_list,
                lambda prompt, text_id: (text_id == delimiter) or len(
                    prompt) > 20,
                lambda prompt: valid_tokens_for_number(
                    vocab, prompt, delimiter))

            if param_type == "number" and '.' not in generated_number:

                input_ids_list.extend(encode_prompt(model, ".0"))

        elif param_type == "string":
            # abre comillas
            input_ids_list.extend(encode_prompt(model, '"'))

            generate_tokens(
                model,
                input_ids_list,
                lambda prompt, text_id: (text_id == '"') or len(
                    prompt) > 20,
                lambda prompt: valid_tokens_for_string(vocab, prompt),
                invalid_token_for_string)

            input_ids_list.extend(encode_prompt(model, '"'))

        elif param_type == "boolean":
            generate_tokens(
                model,
                input_ids_list,
                lambda prompt, text_id: prompt in ("true", "false"),
                lambda prompt: valid_tokens_for_boolean(vocab)
            )

        # hasta aquí - {"prompt": <prompt>, "name": "<generated_name>",
        # "parameters": { <param_name>: <param_value>

        if i < len(params) - 1:
            input_ids_list.extend(encode_prompt(model, ', '))
        else:
            input_ids_list.extend(encode_prompt(model, '}}'))

    return str(model.decode(input_ids_list[initial_input_size:]))
