from .decoder import make_prompt, encode_prompt, constrained_decoding
from .models import FunctionCall, FunctionDef
from .decoder import get_vocab
from llm_sdk import Small_LLM_Model  # type: ignore


def generator(function_defs: list[FunctionDef],
              function_calls: list[FunctionCall]) -> list[str]:
    """
    Ejecuta el pipeline completo de generación: carga el vocabulario,
    arma los prompts y corre la decodificación restringida para cada
    llamada.

    Parameters
    ----------
    function_defs : list[FunctionDef]
        Lista de definiciones de funciones válidas.
    function_calls : list[FunctionCall]
        Lista de llamadas/prompts del usuario.

    Returns
    -------
    list[str]
        Lista de respuestas generadas por la LLM, una cadena JSON por
        prompt.
    """

    prompts = make_prompt(function_defs, function_calls)

    model = Small_LLM_Model()

    vocab, _ = get_vocab(model)

    results = []

    for prompt, call in zip(prompts, function_calls):

        prompt_ids = encode_prompt(model, prompt)

        result_json = constrained_decoding(
            model,
            vocab,
            prompt_ids,
            function_defs,
            call.prompt
        )

        results.append(result_json)

    return results
