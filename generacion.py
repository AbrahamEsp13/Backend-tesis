"""Generación (la "G" de RAG): del contexto recuperado a preguntas validadas.

- Pide a Gemini una salida JSON con esquema fijo (response_schema), no solo "texto en JSON".
- Valida cada pregunta: respuesta_correcta debe ser EXACTAMENTE una de las opciones
  (la calificación compara textos). Si el modelo respondió con una letra ("b", "b)")
  o con el texto sin la letra, se corrige; si no se puede, la pregunta se descarta.
- Cada pregunta indica de qué fragmento del contexto salió y una cita textual. `agregar_fuentes`
  comprueba que la cita exista en el documento y guarda `fuente: {texto, pagina, pagina_fin, literal}`,
  que el estudiante ve después de entregar ("Revisa esto en tu material").
"""
import json
import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()
# El modo JSON (response_mime_type / response_schema) solo existe en la API v1beta;
# las versiones recientes de google-genai usan v1 por defecto y responden 400 sin esto.
_cliente = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version="v1beta"),
)
MODELO = "gemini-2.5-flash"
# Nivel de complejidad elegido por el docente -> indicaciones para el prompt.
COMPLEJIDADES = {
    "basico": (
        "Nivel BÁSICO: preguntas directas sobre definiciones, datos, conceptos y ejemplos que aparecen "
        "explícitamente en el texto. Redacción breve y clara. Los distractores deben ser incorrectos "
        "para quien leyó el material, sin trampas ni matices finos."
    ),
    "equilibrado": (
        "Nivel EQUILIBRADO: combina preguntas sobre conceptos y definiciones con otras que pidan "
        "explicar, aplicar o relacionar ideas del texto."
    ),
    "avanzado": (
        "Nivel AVANZADO: evita preguntas de simple memorización. Plantea situaciones o casos breves que exijan "
        "aplicar, comparar, inferir causas/consecuencias o juzgar la mejor opción con base en el texto. "
        "Los distractores deben ser plausibles y reflejar errores conceptuales comunes."
    ),
}
COMPLEJIDAD_PREDETERMINADA = "equilibrado"


class PreguntaIA(BaseModel):
    pregunta: str
    opciones: list[str]
    respuesta_correcta: str
    justificacion_pedagogica: str
    fragmento: int
    cita: str


class RespuestaInvalida(ValueError):
    """El modelo no devolvió ninguna pregunta utilizable."""


def construir_prompt(contexto: str, num_preguntas: int, complejidad: str = COMPLEJIDAD_PREDETERMINADA) -> str:
    indicaciones = COMPLEJIDADES.get(complejidad, COMPLEJIDADES[COMPLEJIDAD_PREDETERMINADA])
    return f"""Eres un profesor experto en pedagogía. Genera una evaluación basada ÚNICAMENTE en el contexto proporcionado. No inventes información.

Genera exactamente {num_preguntas} preguntas de opción múltiple.
{indicaciones}
Reparte las preguntas entre los distintos fragmentos del contexto para cubrir todo el material.

Reglas para cada pregunta:
- "opciones": exactamente 4 opciones con el formato "a) ...", "b) ...", "c) ...", "d) ...".
- "respuesta_correcta": copia EXACTA, carácter por carácter, de una de las 4 opciones (incluida la letra).
- "justificacion_pedagogica": por qué es correcta, basada en el texto.
- "fragmento": el número N del [Fragmento N] del que sale la pregunta.
- "cita": una o dos oraciones copiadas TEXTUALMENTE de ese fragmento (sin cambiar ni una palabra) que respaldan la respuesta correcta.

Contexto del PDF:
{contexto}
"""


def _sin_letra(texto: str) -> str:
    """'b) París' -> 'parís' (minúsculas, sin la letra inicial)."""
    return re.sub(r"^\s*[a-dA-D]\s*[\).:-]\s*", "", texto).strip().lower()


def resolver_respuesta(respuesta: str, opciones: list[str]) -> str | None:
    """La opción exacta que corresponde a `respuesta`, o None si no se puede determinar."""
    if respuesta in opciones:
        return respuesta
    limpia = respuesta.strip()
    # Solo la letra: "b", "B", "b)", "(b)"
    letra = re.fullmatch(r"\(?\s*([a-dA-D])\s*[\).:]?\s*", limpia)
    if letra:
        coincidencias = [op for op in opciones if re.match(rf"^\s*{letra.group(1).lower()}\s*[\).:-]", op.lower())]
        return coincidencias[0] if len(coincidencias) == 1 else None
    # El mismo texto, ignorando la letra, mayúsculas y espacios
    coincidencias = [op for op in opciones if _sin_letra(op) == _sin_letra(limpia)]
    return coincidencias[0] if len(coincidencias) == 1 else None


def validar_preguntas(datos, num_preguntas: int) -> list[dict]:
    """Normaliza la salida del modelo y descarta las preguntas inválidas."""
    if isinstance(datos, dict):  # p. ej. {"preguntas": [...]}
        listas = [v for v in datos.values() if isinstance(v, list)]
        datos = listas[0] if len(listas) == 1 else []
    if not isinstance(datos, list):
        datos = []

    validas = []
    for item in datos:
        if not isinstance(item, dict):
            continue
        pregunta = str(item.get("pregunta", "")).strip()
        opciones = [str(op).strip() for op in item.get("opciones", []) if str(op).strip()]
        if not pregunta or len(opciones) < 2 or len(set(opciones)) != len(opciones):
            continue
        correcta = resolver_respuesta(str(item.get("respuesta_correcta", "")), opciones)
        if correcta is None:
            continue
        validas.append({
            "pregunta": pregunta,
            "opciones": opciones,
            "respuesta_correcta": correcta,
            "justificacion_pedagogica": str(item.get("justificacion_pedagogica", "")).strip(),
            "puntuacion": 1,
            # Datos crudos del origen; agregar_fuentes los convierte en "fuente" y los quita
            "fragmento": item.get("fragmento"),
            "cita": str(item.get("cita") or "").strip(),
        })

    if not validas:
        raise RespuestaInvalida("La IA no devolvió preguntas válidas. Intenta de nuevo.")
    return validas[:num_preguntas]


LARGO_MINIMO_CITA = 15


def _comparable(texto: str) -> str:
    """Minúsculas y espacios colapsados (el PDF parte líneas donde el modelo no lo hace)."""
    return re.sub(r"\s+", " ", texto).strip().lower()


def _limpiar_cita(cita: str) -> str:
    """Quita comillas y puntos suspensivos con que el modelo a veces envuelve la cita."""
    return cita.strip().strip("\"'«»“”‘’").strip().removeprefix("...").removesuffix("...").strip()


def _numero(valor) -> int | None:
    try:
        return int(valor)
    except (TypeError, ValueError):
        return None


def buscar_fuente(numero, cita: str, fragmentos: list[dict]) -> dict | None:
    """
    La fuente de una pregunta a partir del fragmento y la cita que indicó el modelo.
    - La cita se busca primero en el fragmento indicado y luego en los demás
      (el modelo a veces se equivoca de número pero copia bien el texto).
    - Si la cita no aparece en ningún fragmento, se usa el fragmento indicado completo.
    - Si tampoco el número es válido, no hay fuente (None).
    """
    por_numero = {f["numero"]: f for f in fragmentos}
    indicado = por_numero.get(_numero(numero))
    cita = _limpiar_cita(cita)
    if len(cita) >= LARGO_MINIMO_CITA:
        buscada = _comparable(cita)
        candidatos = ([indicado] if indicado else []) + [f for f in fragmentos if f is not indicado]
        for fragmento in candidatos:
            if buscada in _comparable(fragmento["texto"]):
                return {"texto": cita, "pagina": fragmento["pagina"], "pagina_fin": fragmento["pagina_fin"], "literal": True}
    if indicado:
        return {"texto": indicado["texto"], "pagina": indicado["pagina"], "pagina_fin": indicado["pagina_fin"], "literal": False}
    return None


def agregar_fuentes(preguntas: list[dict], fragmentos: list[dict]) -> list[dict]:
    """Reemplaza los campos crudos `fragmento` y `cita` por `fuente` (cuando se pudo determinar)."""
    resultado = []
    for pregunta in preguntas:
        pregunta = dict(pregunta)
        fuente = buscar_fuente(pregunta.pop("fragmento", None), pregunta.pop("cita", "") or "", fragmentos)
        if fuente:
            pregunta["fuente"] = fuente
        resultado.append(pregunta)
    return resultado


def generar_preguntas(contexto: str, num_preguntas: int, complejidad: str = COMPLEJIDAD_PREDETERMINADA) -> list[dict]:
    respuesta = _cliente.models.generate_content(
        model=MODELO,
        contents=construir_prompt(contexto, num_preguntas, complejidad),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[PreguntaIA],
            temperature=0.2,
        ),
    )
    try:
        datos = json.loads(respuesta.text)
    except (TypeError, ValueError):
        raise RespuestaInvalida("La IA devolvió una respuesta con formato inválido. Intenta de nuevo.")
    return validar_preguntas(datos, num_preguntas)
