"""Calificación de intentos en el servidor.

El estudiante nunca recibe las respuestas correctas antes de entregar: el backend
califica con las preguntas guardadas y solo entonces devuelve la corrección.
"""

# Campos que el estudiante no debe ver mientras resuelve el examen
CAMPOS_PRIVADOS = ("respuesta_correcta", "justificacion_pedagogica", "fuente")


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


def calificar(preguntas: list, respuestas: dict) -> dict:
    """
    Califica un intento.
    `respuestas` es { "<índice de pregunta>": "<opción elegida>" } (las claves llegan como texto en JSON).
    """
    obtenido = 0.0
    total = 0.0
    respuestas_correctas = []
    justificaciones = []
    fuentes = []
    for indice, pregunta in enumerate(preguntas):
        puntos = puntos_de_pregunta(pregunta)
        total += puntos
        correcta = pregunta.get("respuesta_correcta")
        respuestas_correctas.append(correcta)
        justificaciones.append(pregunta.get("justificacion_pedagogica") or "")
        fuentes.append(pregunta.get("fuente"))
        if respuestas.get(str(indice)) == correcta:
            obtenido += puntos

    calificacion = round(obtenido / total * 10, 2) if total > 0 else 0.0
    return {
        "puntos_obtenidos": obtenido,
        "puntos_totales": total,
        "calificacion": calificacion,
        "respuestas_correctas": respuestas_correctas,
        "justificaciones": justificaciones,  # retroalimentación que ve el estudiante tras entregar
        "fuentes": fuentes,  # {texto, pagina, pagina_fin} o None: de qué parte del PDF salió cada pregunta
    }
