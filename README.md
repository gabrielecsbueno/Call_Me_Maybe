# **Call Me Maybe**

---

# 1. Descripcion

El proyecto `Call_Me_Maybe` es una implementación de **function calling** guiada por LLM: dado un conjunto de funciones disponibles y una pregunta en lenguaje natural, el programa decide qué función corresponde y genera los argumentos correctos, devolviendo el resultado como un objeto JSON válido.

El desafío central es que los LLM pequeños (este proyecto usa Qwen3-0.6B) no son confiables generando salidas estructuradas solo con prompting. Este proyecto resuelve eso con **decodificación restringida** (*constrained decoding*), garantizando JSON 100% válido por construcción, no por probabilidad.

## 1.1. El algoritmo

La generación no confía en que el LLM "elija" producir JSON válido por su cuenta. En cambio, el texto se construye token por token y, en cada paso, se restringe qué tokens puede elegir el modelo — de ahí el nombre *decodificación restringida* (constrained decoding).

**Mecánica básica** (`generate_tokens` en `decoder.py`):
- en cada iteración se piden al modelo los logits para el próximo token dada la secuencia acumulada;
- se pone `-inf` en todo id que no sea válido en ese punto (según la función de validación del campo que se está generando);
- se elige el token de mayor logit restante (`argmax`, sin sampling — determinístico, y suficiente porque ya solo quedan opciones válidas);
- se decodifica ese id a texto;
- y se comprueba si ese token es la señal de que el campo terminó.

**Construcción campo por campo:** el JSON de salida no se genera de una sola vez. Las partes fijas del template (`{"prompt": `, `"name": "`, `, "parameters": {`, comas, llaves de cierre) se inyectan directamente como texto conocido, tokenizado normalmente. Solo las partes variables pasan por el loop restringido:

- **Nombre de función**: el conjunto de nombres válidos es finito y conocido de antemano (viene de `functions_definition.json`). Un token es válido si, sumado a lo ya generado, sigue siendo prefijo de al menos un nombre real. La generación para cuando el texto acumulado ya es exactamente igual a un nombre completo — el modelo "confirma" con un token extra que se descarta, sin contarse como parte del nombre.

- **Número**: son válidos los dígitos, el punto y el signo negativo, respetando lo que ya se generó (no se permite un segundo punto decimal, ni un signo que no esté al principio). La generación para cuando el modelo elige el delimitador esperado a continuación en el template (`,` o `}`), que tampoco se cuenta como parte del valor.

- **Strings**: son válidos todos los tokens que no contengan comillas sin escapar, barra invertida ni caracteres de control — la comilla de cierre en sí también es un token válido, y es la señal de parada.

- **Booleano**: son válidos los tokens `"true"` o `"false"`. La generación para cuando el texto coincide exactamente con uno de los dos valores.

## 1.2. Decisiones de diseño

Cada pregunta pasa por las mismas etapas independientes y en el mismo orden, así que organizar el proceso como un pipeline facilita razonar sobre cada paso por separado y detectar en qué punto falla algo.

**Pipeline:** :
1. cargar funciones y preguntas desde JSON
2. construir un prompt por pregunta
3. tokenizar
4. generar cada respuesta con decodificación restringida
5. combinar y guardar los resultados.

**Módulos:** :
`models.py`     (contrato de datos, Pydantic),
`loader.py`     (I/O y validación),
`decoder.py`    (motor de decodificación restringida),
`generator.py`  (orquesta el pipeline),
`__main__.py`   (CLI).

**Otras decisiones:**
- Validación con Pydantic al cargar los datos, no dicts sueltos en el código.

- Generación incremental campo por campo, en vez de generar todo y validar después — así el JSON es válido por construcción.

- `try/except` específico en cada operación de archivo, con una captura general en `main()` para que ninguna excepción llegue a romper el programa.

## 1.3. Analisis de rendimiento

Al restringir los tokens válidos en cada paso, la estructura del JSON de salida es válida por construcción — no depende de que el modelo "acierte" el formato. Lo que sigue sin garantía es la corrección semántica (elegir la función correcta, los valores correctos): eso depende de qué tan bien el modelo entiende la pregunta, y con un modelo tan pequeno como Qwen3-0.6B puede fallar en casos ambiguos o con contenido de texto libre más largo o complejo.

## 1.4. Retos encontrados

El principal desafío fue el propio algoritmo: para hacerlo más rápido haría falta codificar de antemano cada token del vocabulario de la LLM, en vez de recalcular el filtro sobre miles de tokens en cada paso de la generación.

## 1.5. Estrategia de pruebas

La validación fue manual: correr el pipeline completo (`uv run python -m src`) contra
los datos reales (`functions_definition.json` + `function_calling_tests.json`, 11 preguntas) e inspeccionar la salida.


## 1.6. Ejemplos de uso

```bash
uv run python -m src
```

Con los datos de ejemplo (`data/input/`), la salida
(`data/output/function_calling_results.json`) incluye resultados como:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}
{"prompt": "Greet shrek", "name": "fn_greet", "parameters": {"name": "shrek"}}
{"prompt": "Reverse the string 'hello'", "name": "fn_reverse_string", "parameters": {"s": "hello"}}
```

Para usar otros archivos:

```bash
uv run python -m src --functions_definition <ruta/funciones.json> --input <ruta/preguntas.json> --output <ruta/salida.json>
```

---

# Estructura del proyecto

```bash
call_me_maybe/
├── data/
│   ├── input/
│   │   ├── functions_definition.json
│   │   └── function_calling_tests.json
│   └── output/              # generado por el programa
├── src/
│   ├── __init__.py
│   ├── __main__.py          # punto de entrada: lee argumentos y llama a main()
│   ├── models.py            # modelos Pydantic
│   ├── loader.py            # carga los JSON de entrada
│   ├── decoder.py           # decodificacion restringida
│   └── generator.py         # pipeline de generacion
├── llm_sdk/                 # paquete de la LLM
│   ├── llm_sdk/
│   │   └── __init__.py
│   ├── pyproject.toml
│   └── uv.lock
├── pyproject.toml
├── uv.lock
├── Makefile
├── README.md
└── .gitignore
```

---

# 2. Instrucciones

## Requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

## Instalación

```bash
make install
```

O:

```bash
uv sync
```

## Ejecución

```bash
make run
```

O:

```bash
uv run python -m src
```

Por defecto usa `data/input/functions_definition.json` y `data/input/function_calling_tests.json`, y guarda el resultado en `data/output/function_calling_results.json`. Se pueden indicar otros archivos:

```bash
uv run python -m src --functions_definition <ruta> --input <ruta> --output <ruta>
```

## Otros comandos del Makefile

- `make debug`: corre el programa con el depurador `pdb`.
- `make clean`: borra `__pycache__`, `.mypy_cache` y `.venv`.
- `make lint` : corre `flake8` y `mypy`.

---

# 3. Recursos

# 3.1. Documentación y enlaces útiles.

[Entendendo CLI](https://www.alura.com.br/artigos/cli-interface-linha-comandos)

[Como criar uma CLI em Python](https://ellibrodepython.com/python-argparse)

[Python e argparse](https://www.youtube.com/watch?v=Ad6934NXn4A)

[Modulo Argparse - Doc](https://www.w3schools.com/python/ref_module_argparse.asp)

[Argparse - Documentacao](https://docs.python.org/3/library/argparse.html)

[Arquivos JSON](https://www.alura.com.br/artigos/o-que-e-json?srsltid=AfmBOoqAoGmQtrxUNs663GWthkX40rX66oY9JYEo9bswtxR38cNHK2na)

[UV Python](https://hub.asimov.academy/blog/uv-python/)

[O que e LLM e como funciona](https://www.ibm.com/br-pt/think/topics/large-language-models?regionCode=br&languageCode=pt&cm-history=br-pt)

[Aprendizado Autossupervisionado](https://www.ibm.com/br-pt/think/topics/self-supervised-learning?regionCode=br&languageCode=pt&cm-history=br-pt)

[Documentacao da Qwen3-0.6 da HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B)
