"""Generación (la "G" de RAG): del contexto recuperado a preguntas validadas.

- Pide a Gemini una salida JSON con esquema fijo (response_schema), no solo "texto en JSON".
- Valida cada pregunta: respuesta_correcta debe ser EXACTAMENTE una de las opciones
  (la calificación compara textos). Si el modelo respondió con una letra ("b", "b)")
  o con el texto sin la letra, se corrige; si no se puede, la pregunta se descarta.
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
NIVELES_BLOOM = ["Recordar", "Comprender", "Aplicar", "Analizar", "Evaluar", "Crear"]

# Nivel de complejidad elegido por el docente -> niveles de Bloom e indicaciones para el prompt.
# "Crear" queda fuera de "avanzado": producir algo nuevo no se mide bien con opción múltiple.
COMPLEJIDADES = {
    "basico": {
        "niveles": ["Recordar", "Comprender"],
        "indicaciones": (
            "Nivel BÁSICO: preguntas directas sobre definiciones, datos, conceptos y ejemplos que aparecen "
            "explícitamente en el texto. Redacción breve y clara. Los distractores deben ser incorrectos "
            "para quien leyó el material, sin trampas ni matices finos."
        ),
    },
    "equilibrado": {
        "niveles": NIVELES_BLOOM,
        "indicaciones": (
            "Nivel EQUILIBRADO: combina preguntas de memoria y comprensión con otras de aplicación y análisis, "
            "repartidas entre los niveles indicados."
        ),
    },
    "avanzado": {
        "niveles": ["Aplicar", "Analizar", "Evaluar"],
        "indicaciones": (
            "Nivel AVANZADO: evita preguntas de simple memorización. Plantea situaciones o casos breves que exijan "
            "aplicar, comparar, inferir causas/consecuencias o juzgar la mejor opción con base en el texto. "
            "Los distractores deben ser plausibles y reflejar errores conceptuales comunes."
        ),
    },
}
COMPLEJIDAD_PREDETERMINADA = "equilibrado"


class PreguntaIA(BaseModel):
    nivel_bloom: str
    pregunta: str
    opciones: list[str]
    respuesta_correcta: str
    justificacion_pedagogica: str


class RespuestaInvalida(ValueError):
    """El modelo no devolvió ninguna pregunta utilizable."""


def construir_prompt(contexto: str, num_preguntas: int, complejidad: str = COMPLEJIDAD_PREDETERMINADA) -> str:
    nivel = COMPLEJIDADES.get(complejidad, COMPLEJIDADES[COMPLEJIDAD_PREDETERMINADA])
    niveles = ", ".join(nivel["niveles"])
    return f"""Eres un profesor experto en pedagogía. Genera una evaluación basada ÚNICAMENTE en el contexto proporcionado. No inventes información.

Genera exactamente {num_preguntas} preguntas de opción múltiple, distribuidas en los niveles de la Taxonomía de Bloom ({niveles}).
{nivel["indicaciones"]}
Reparte las preguntas entre los distintos fragmentos del contexto para cubrir todo el material.

Reglas para cada pregunta:
- "opciones": exactamente 4 opciones con el formato "a) ...", "b) ...", "c) ...", "d) ...".
- "respuesta_correcta": copia EXACTA, carácter por carácter, de una de las 4 opciones (incluida la letra).
- "justificacion_pedagogica": por qué es correcta, basada en el texto.
- "nivel_bloom": uno de {niveles}.

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
            "nivel_bloom": str(item.get("nivel_bloom", "")).strip() or "Comprender",
            "pregunta": pregunta,
            "opciones": opciones,
            "respuesta_correcta": correcta,
            "justificacion_pedagogica": str(item.get("justificacion_pedagogica", "")).strip(),
            "puntuacion": 1,
        })

    if not validas:
        raise RespuestaInvalida("La IA no devolvió preguntas válidas. Intenta de nuevo.")
    return validas[:num_preguntas]


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
