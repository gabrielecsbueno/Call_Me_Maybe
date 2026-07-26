from .loader import load_functions, load_calls, make_json
from .generator import generator
import argparse
from pydantic import ValidationError


def parse_args() -> argparse.Namespace:
    """
    Analiza los argumentos ingresados por el usuario en la línea de
    comandos.

    Returns
    -------
    argparse.Namespace
        El objeto Namespace que devuelve parse_args(), que guarda los
        valores de los argumentos de la línea de comandos.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json",
                        type=str,
                        help="Archivo de definición de funciones")

    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json",
                        type=str, help="Archivo de entrada")

    parser.add_argument("--output",
                        default="data/output/function_calling_results.json",
                        type=str, help="Archivo de salida")

    return parser.parse_args()


def main() -> None:
    """
    Punto de entrada del programa: lee los archivos JSON de funciones y
    prompts, corre el pipeline de generación vía LLM y guarda el
    resultado en JSON.

    Los errores de validación (Pydantic) o cualquier otra excepción
    generada en cualquier etapa del pipeline se capturan y se imprimen,
    sin propagarse.

    Returns
    -------
    None
    """
    args = parse_args()
    try:
        function_defs = load_functions(
            args.functions_definition
        )

        function_calls = load_calls(args.input)

        results = generator(function_defs, function_calls)

        make_json(results, args.output)

    except ValidationError as exc:

        print("Error: ValidationError", end='')
        for error in exc.errors():
            error_message = error['msg']
            for error_location in error['loc']:
                print(f" ({error_location} = {error_message})")

    except Exception as err:
        print("Error:", err)

    else:
        print("\n¡JSON generado con éxito!")
        print(f"Archivo guardado en {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nError: salida interrumpida inesperadamente.")
