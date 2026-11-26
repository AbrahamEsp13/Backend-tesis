"""2.3 Práctica adaptativa: contexto con los fragmentos fallados, límite diario y privacidad."""
import pytest

import practicas
from database import Practica

PREGUNTAS = [
    {"pregunta": "p1", "opciones": ["a", "b"], "respuesta_correcta": "a", "justificacion_pedagogica": "j1",
     "fuente": {"texto": "cita uno", "pagina": 2, "pagina_fin": 2, "literal": True, "fragmento": "Fragmento completo uno."}},
    {"pregunta": "p2", "opciones": ["a", "b"], "respuesta_correcta": "b", "justificacion_pedagogica": "j2",
     "fuente": {"texto": "Fragmento sin cita literal.", "pagina": 5, "pagina_fin": 6, "literal": False}},
    {"pregunta": "p3", "opciones": ["a", "b"], "respuesta_correcta": "a", "justificacion_pedagogica": "j3"},
    {"pregunta": "p4", "opciones": ["a", "b"], "respuesta_correcta": "b", "justificacion_pedagogica": "j4",
     "fuente": {"texto": "cita uno", "pagina": 2, "pagina_fin": 2, "literal": True, "fragmento": "Fragmento completo uno."}},
]


# ---------- Lógica ----------

def test_preguntas_falladas_incluye_las_no_respondidas():
    assert practicas.preguntas_falladas(PREGUNTAS, {"0": "a", "1": "a", "3": "b"}) == [1, 2]


def test_numero_de_preguntas_entre_minimo_y_maximo():
    assert practicas.numero_de_preguntas(1) == practicas.MIN_PREGUNTAS
    assert practicas.numero_de_preguntas(3) == 6
    assert practicas.numero_de_preguntas(20) == practicas.MAX_PREGUNTAS


def test_fragmentos_usan_el_fragmento_completo_y_no_se_repiten():
    fragmentos = practicas.fragmentos_de_errores(PREGUNTAS, [0, 1, 2, 3])
    assert [f["numero"] for f in fragmentos] == [1, 2, 3]
    assert fragmentos[0] == {"numero": 1, "texto": "Fragmento completo uno.", "pagina": 2, "pagina_fin": 2}
    assert fragmentos[1]["texto"] == "Fragmento sin cita literal." and fragmentos[1]["pagina"] == 5
    # Sin fuente: la pregunta con su clave y justificación, sin página
    assert fragmentos[2]["pagina"] is None
    assert "p3" in fragmentos[2]["texto"] and "Respuesta correcta: a" in fragmentos[2]["texto"]


def test_fuentes_sin_pagina_se_descartan():
    fragmentos = practicas.fragmentos_de_errores(PREGUNTAS, [0, 2])
    nuevas = [
        {"pregunta": "n1", "opciones": ["a", "b"], "respuesta_correcta": "a", "fragmento": 1, "cita": ""},
        {"pregunta": "n2", "opciones": ["a", "b"], "respuesta_correcta": "a", "fragmento": 2, "cita": ""},
    ]
    resultado = practicas.con_fuentes_validas(nuevas, fragmentos)
    assert resultado[0]["fuente"]["pagina"] == 2
    assert "fuente" not in resultado[1]
    assert all("fragmento" not in p and "cita" not in p for p in resultado)


# ---------- Endpoints ----------

@pytest.fixture
def generador(monkeypatch):
    """Reemplaza a Gemini: registra el contexto y los errores con que se llamó."""
    llamadas = []

    def generar(contexto, num_preguntas, errores):
        llamadas.append({"contexto": contexto, "num_preguntas": num_preguntas, "errores": errores})
        return [{"pregunta": f"nueva {i}", "opciones": ["a) x", "b) y"], "respuesta_correcta": "a) x",
                 "justificacion_pedagogica": "j", "puntuacion": 1, "fragmento": 1, "cita": ""}
                for i in range(num_preguntas)]
    monkeypatch.setattr(practicas, "generar_practica", generar)
    return llamadas


def entregar(cliente, datos, respuestas):
    respuesta = cliente.post(f"/api/cuestionarios/{datos['examen_id']}/intentos", headers=datos["estudiante"],
                             json={"respuestas": respuestas})
    assert respuesta.status_code == 200, respuesta.text
    return respuesta.json()["intento"]["id"]


def test_crear_practica_con_lo_fallado(cliente, clase_con_examen, generador):
    intento_id = entregar(cliente, clase_con_examen, {"0": "a", "1": "a"})  # falla la 2
    respuesta = cliente.post(f"/api/intentos/{intento_id}/practica", headers=clase_con_examen["estudiante"])
    assert respuesta.status_code == 200, respuesta.text
    datos = respuesta.json()
    assert datos["restantes"] == practicas.PRACTICAS_DIARIAS - 1
    practica = datos["practica"]
    assert practica["preguntas_origen"] == [1]
    assert len(practica["preguntas_json"]) == practicas.MIN_PREGUNTAS
    # La práctica trae la clave (se corrige al instante en el navegador)
    assert practica["preguntas_json"][0]["respuesta_correcta"] == "a) x"
    # Solo el error entra al prompt
    assert generador[0]["errores"] == [{"pregunta": "p2", "respuesta_correcta": "b", "elegida": "a"}]
    assert "p2" in generador[0]["contexto"] and "p1" not in generador[0]["contexto"]


def test_sin_errores_no_hay_practica(cliente, clase_con_examen, generador):
    intento_id = entregar(cliente, clase_con_examen, {"0": "a", "1": "b"})
    respuesta = cliente.post(f"/api/intentos/{intento_id}/practica", headers=clase_con_examen["estudiante"])
    assert respuesta.status_code == 400
    assert generador == []


def test_limite_diario(cliente, clase_con_examen, generador):
    intento_id = entregar(cliente, clase_con_examen, {})
    estudiante = clase_con_examen["estudiante"]
    for _ in range(practicas.PRACTICAS_DIARIAS):
        assert cliente.post(f"/api/intentos/{intento_id}/practica", headers=estudiante).status_code == 200
    respuesta = cliente.post(f"/api/intentos/{intento_id}/practica", headers=estudiante)
    assert respuesta.status_code == 429
    assert cliente.get("/api/practicas/cupo", headers=estudiante).json() == {"restantes": 0, "limite": practicas.PRACTICAS_DIARIAS}
    assert len(generador) == practicas.PRACTICAS_DIARIAS


def test_solo_el_dueno_del_intento(cliente, registrar, clase_con_examen, generador):
    intento_id = entregar(cliente, clase_con_examen, {})
    otro, _ = registrar("beto", "estudiante")
    assert cliente.post(f"/api/intentos/{intento_id}/practica", headers=otro).status_code == 404
    assert cliente.post(f"/api/intentos/{intento_id}/practica", headers=clase_con_examen["docente"]).status_code == 403
    practica_id = cliente.post(f"/api/intentos/{intento_id}/practica", headers=clase_con_examen["estudiante"]).json()["practica"]["id"]
    assert cliente.get(f"/api/practicas/{practica_id}", headers=otro).status_code == 404


def test_terminar_practica_y_conteo_del_docente(cliente, clase_con_examen, generador, db):
    intento_id = entregar(cliente, clase_con_examen, {})
    estudiante = clase_con_examen["estudiante"]
    practica = cliente.post(f"/api/intentos/{intento_id}/practica", headers=estudiante).json()["practica"]
    ruta = f"/api/practicas/{practica['id']}/respuestas"

    # Aún sin terminar: el docente no la cuenta
    resultados = cliente.get(f"/api/cuestionarios/{clase_con_examen['examen_id']}/intentos", headers=clase_con_examen["docente"]).json()
    assert resultados["practicas"] == {}

    terminada = cliente.put(ruta, headers=estudiante, json={"respuestas": {"0": "a) x", "1": "b) y", "99": "a) x"}}).json()["practica"]
    assert terminada["aciertos"] == 1  # 1 acierto de 3 (la 3 sin responder)
    assert terminada["respuestas"] == {"0": "a) x", "1": "b) y"}
    assert terminada["fecha_terminada"] is not None

    # Una práctica terminada no cambia
    otra_vez = cliente.put(ruta, headers=estudiante, json={"respuestas": {"0": "a) x", "1": "a) x", "2": "a) x"}}).json()["practica"]
    assert otra_vez["aciertos"] == 1

    resultados = cliente.get(f"/api/cuestionarios/{clase_con_examen['examen_id']}/intentos", headers=clase_con_examen["docente"]).json()
    estudiante_id = str(clase_con_examen["usuario_estudiante"]["id"])
    assert resultados["practicas"] == {estudiante_id: 1}


def test_borrar_el_examen_borra_sus_practicas(cliente, clase_con_examen, generador, db):
    intento_id = entregar(cliente, clase_con_examen, {})
    cliente.post(f"/api/intentos/{intento_id}/practica", headers=clase_con_examen["estudiante"])
    assert db.query(Practica).count() == 1
    respuesta = cliente.delete(f"/api/cuestionarios/{clase_con_examen['examen_id']}", headers=clase_con_examen["docente"])
    assert respuesta.status_code == 200
    db.expire_all()
    assert db.query(Practica).count() == 0
