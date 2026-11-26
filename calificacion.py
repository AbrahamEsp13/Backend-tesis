"""Calificación de intentos en el servidor.

El estudiante nunca recibe las respuestas correctas antes de entregar: el backend
califica con las preguntas guardadas y solo entonces devuelve la corrección.
Las respuestas son ids de opción (ver identificadores.py), no textos.

Tipos de pregunta (4.2):
- "opcion_multiple" (o sin `tipo`, las anteriores): se califica por id de opción.
- "verdadero_falso": igual que la opción múltiple, con las opciones "Verdadero" y "Falso".
- "respuesta_corta": el estudiante escribe; `respuesta_correcta` es la respuesta modelo. La califica
  la IA (respuestas_cortas.py) con 1, 0.5 o 0; el docente puede cambiarla. El resultado se guarda
  aparte en el intento (`respuestas_cortas`) y aquí solo se suman sus puntos.
"""
from identificadores import es_correcta

OPCION_MULTIPLE = "opcion_multiple"
VERDADERO_FALSO = "verdadero_falso"
RESPUESTA_CORTA = "respuesta_corta"
TIPOS = (OPCION_MULTIPLE, VERDADERO_FALSO, RESPUESTA_CORTA)
OPCIONES_VERDADERO_FALSO = ["Verdadero", "Falso"]

# Campos que el estudiante no debe ver mientras resuelve el examen
# (la valoración del docente no es secreta, pero no le sirve al estudiante)
CAMPOS_PRIVADOS = ("respuesta_correcta", "id_correcta", "justificacion_pedagogica", "fuente", "valoracion")


def tipo_de(pregunta: dict) -> str:
    tipo = pregunta.get("tipo")
    return tipo if tipo in TIPOS else OPCION_MULTIPLE


def es_respuesta_corta(pregunta: dict) -> bool:
    return tipo_de(pregunta) == RESPUESTA_CORTA


def puntos_de_pregunta(pregunta: dict) -> float:
    """Puntos de una pregunta; si no tiene puntuación válida, vale 1 (igual que el frontend)."""
    try:
        puntos = float(pregunta.get("puntuacion") or 0)
    except (TypeError, ValueError):
        puntos = 0
    return puntos if puntos > 0 else 1


def ocultar_respuestas(preguntas: list) -> list:
    """Copia de las preguntas sin la clave ni la justificación (para estudiantes)."""
    return [
        {campo: valor for campo, valor in pregunta.items() if campo not in CAMPOS_PRIVADOS}
        for pregunta in preguntas
    ]


def fraccion_de_acierto(pregunta: dict, valor, corta: dict | None = None) -> float:
    """De 0 a 1: cuánto vale la respuesta (las de opción son 0 o 1; las cortas, lo que se calificó)."""
    if es_respuesta_corta(pregunta):
        try:
            return min(1.0, max(0.0, float((corta or {}).get("puntaje") or 0)))
        except (TypeError, ValueError):
            return 0.0
    return 1.0 if es_correcta(pregunta, valor) else 0.0


def calificar(preguntas: list, respuestas: dict, cortas: dict | None = None) -> dict:
    """
    Califica un intento.
    `respuestas` es { "<índice de pregunta>": "<id de la opción elegida>" (o el texto escrito, en las cortas) }.
    `cortas` es { "<índice>": {"puntaje": 0|0.5|1, ...} } con la calificación de las respuestas cortas.
    En preguntas antiguas sin ids se compara el texto (ver identificadores.es_correcta).
    """
    cortas = cortas or {}
    obtenido = 0.0
    total = 0.0
    respuestas_correctas = []
    justificaciones = []
    fuentes = []
    for indice, pregunta in enumerate(preguntas):
        puntos = puntos_de_pregunta(pregunta)
        total += puntos
        respuestas_correctas.append(pregunta.get("respuesta_correcta"))  # en las cortas, la respuesta modelo
        justificaciones.append(pregunta.get("justificacion_pedagogica") or "")
        fuentes.append(pregunta.get("fuente"))
        obtenido += puntos * fraccion_de_acierto(pregunta, respuestas.get(str(indice)), cortas.get(str(indice)))

    obtenido = round(obtenido, 2)
    calificacion = round(obtenido / total * 10, 2) if total > 0 else 0.0
    return {
        "puntos_obtenidos": obtenido,
        "puntos_totales": total,
        "calificacion": calificacion,
        "respuestas_correctas": respuestas_correctas,
        "justificaciones": justificaciones,  # retroalimentación que ve el estudiante tras entregar
        "fuentes": fuentes,  # {texto, pagina, pagina_fin} o None: de qué parte del PDF salió cada pregunta
    }
