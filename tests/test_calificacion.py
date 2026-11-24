"""Pruebas unitarias de calificacion.py (sin base de datos)."""
from calificacion import calificar, ocultar_respuestas, puntos_de_pregunta

PREGUNTAS = [
    {"pregunta": "a", "opciones": ["x", "y"], "respuesta_correcta": "x", "justificacion_pedagogica": "j", "puntuacion": 2},
    {"pregunta": "b", "opciones": ["x", "y"], "respuesta_correcta": "y", "puntuacion": None},
]


def test_suma_puntos_y_calcula_sobre_10():
    resultado = calificar(PREGUNTAS, {"0": "x", "1": "x"})
    assert resultado["puntos_obtenidos"] == 2
    assert resultado["puntos_totales"] == 3
    assert resultado["calificacion"] == 6.67
    assert resultado["respuestas_correctas"] == ["x", "y"]


def test_sin_respuestas_es_cero():
    assert calificar(PREGUNTAS, {})["calificacion"] == 0


def test_examen_sin_preguntas_no_divide_entre_cero():
    assert calificar([], {})["calificacion"] == 0


def test_puntuacion_invalida_vale_1():
    for valor in (None, 0, -3, "abc", ""):
        assert puntos_de_pregunta({"puntuacion": valor}) == 1
    assert puntos_de_pregunta({"puntuacion": "4"}) == 4


def test_ocultar_respuestas_quita_clave_y_justificacion():
    ocultas = ocultar_respuestas(PREGUNTAS)
    assert all("respuesta_correcta" not in p and "justificacion_pedagogica" not in p for p in ocultas)
    assert ocultas[0]["opciones"] == ["x", "y"]
    assert "respuesta_correcta" in PREGUNTAS[0], "no debe modificar las preguntas originales"


def test_devuelve_justificaciones_en_orden():
    assert calificar(PREGUNTAS, {})["justificaciones"] == ["j", ""]
