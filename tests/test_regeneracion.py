"""4.3 Regenerar una sola pregunta: contexto del fragmento, cupo diario propio y sin gastar créditos."""
import datetime

import pytest

import generacion
import regeneracion
from database import Usuario

FUENTE = {"texto": "cita breve", "pagina": 4, "pagina_fin": 5, "literal": True,
          "fragmento": "Una clase define los atributos y métodos de sus objetos."}
PREGUNTA = {"pregunta": "¿Qué define una clase?", "opciones": ["a) x", "b) y"], "respuesta_correcta": "a) x",
            "justificacion_pedagogica": "j", "puntuacion": 3, "fuente": FUENTE}


@pytest.fixture
def ia(monkeypatch):
    llamadas = []

    def regenerar(contexto, original, complejidad, indicaciones):
        llamadas.append({"contexto": contexto, "original": original, "complejidad": complejidad, "indicaciones": indicaciones})
        return [{"pregunta": "nueva", "opciones": ["a) p", "b) q"], "respuesta_correcta": "b) q",
                 "justificacion_pedagogica": "jn", "puntuacion": 1, "fragmento": 1, "cita": "define los atributos y métodos"}]
    monkeypatch.setattr(regeneracion, "regenerar_pregunta", regenerar)
    return llamadas


@pytest.fixture
def examen(registrar, crear_cuestionario):
    docente, usuario = registrar("docente", "docente")
    return {"docente": docente, "usuario": usuario, "id": crear_cuestionario(usuario["id"], publicado=False)}


def regenerar(cliente, examen, **cuerpo):
    return cliente.post(f"/api/cuestionarios/{examen['id']}/regenerar-pregunta", headers=examen["docente"],
                        json={"pregunta": PREGUNTA, **cuerpo})


def test_regenera_con_el_fragmento_y_conserva_el_puntaje(cliente, examen, ia):
    respuesta = regenerar(cliente, examen, motivo="demasiado_facil", indicaciones="  Usa un ejemplo  ")
    assert respuesta.status_code == 200, respuesta.text
    datos = respuesta.json()
    nueva = datos["pregunta"]
    assert nueva["pregunta"] == "nueva" and nueva["respuesta_correcta"] == "b) q"
    assert nueva["puntuacion"] == 3 and nueva["origen"] == "regenerada"
    assert nueva["fuente"]["pagina"] == 4 and nueva["fuente"]["literal"] is True
    assert "fragmento" not in nueva and "cita" not in nueva  # campos crudos del modelo
    assert datos["regeneraciones_restantes"] == regeneracion.REGENERACIONES_DIARIAS - 1

    llamada = ia[0]
    assert FUENTE["fragmento"] in llamada["contexto"]
    assert llamada["original"]["pregunta"] == PREGUNTA["pregunta"]
    assert "demasiado fácil" in llamada["indicaciones"] and "Usa un ejemplo" in llamada["indicaciones"]


def test_no_gasta_creditos_y_no_guarda_el_examen(cliente, examen, ia, db):
    antes = cliente.get("/api/cuestionarios", headers=examen["docente"]).json()["data"][0]["preguntas_json"]
    regenerar(cliente, examen)
    usuario = db.query(Usuario).filter(Usuario.id == examen["usuario"]["id"]).first()
    assert usuario.creditos_disponibles == 5
    assert cliente.get("/api/cuestionarios", headers=examen["docente"]).json()["data"][0]["preguntas_json"] == antes


def test_pregunta_sin_fuente_usa_su_texto_y_no_inventa_pagina(cliente, examen, ia):
    sin_fuente = {**PREGUNTA, "fuente": None}
    nueva = cliente.post(f"/api/cuestionarios/{examen['id']}/regenerar-pregunta", headers=examen["docente"],
                         json={"pregunta": sin_fuente}).json()["pregunta"]
    assert "fuente" not in nueva
    assert "Respuesta correcta: a) x" in ia[0]["contexto"]


def test_cupo_diario_y_reinicio(cliente, examen, ia, db):
    usuario = db.query(Usuario).filter(Usuario.id == examen["usuario"]["id"]).first()
    usuario.regeneraciones_usadas = regeneracion.REGENERACIONES_DIARIAS
    usuario.fecha_ultimo_uso = datetime.date.today()
    db.commit()
    assert regenerar(cliente, examen).status_code == 429
    assert ia == []

    usuario.fecha_ultimo_uso = datetime.date.today() - datetime.timedelta(days=1)
    db.commit()
    assert regenerar(cliente, examen).json()["regeneraciones_restantes"] == regeneracion.REGENERACIONES_DIARIAS - 1


def test_solo_el_dueno(cliente, examen, registrar, ia):
    otro, _ = registrar("otro", "docente")
    estudiante, _ = registrar("ana", "estudiante")
    ruta = f"/api/cuestionarios/{examen['id']}/regenerar-pregunta"
    assert cliente.post(ruta, headers=otro, json={"pregunta": PREGUNTA}).status_code in (403, 404)
    assert cliente.post(ruta, headers=estudiante, json={"pregunta": PREGUNTA}).status_code == 403


def test_error_de_la_ia_no_consume_cupo(cliente, examen, monkeypatch, db):
    def falla(*_):
        raise generacion.RespuestaInvalida("La IA no devolvió preguntas válidas. Intenta de nuevo.")
    monkeypatch.setattr(regeneracion, "regenerar_pregunta", falla)
    assert regenerar(cliente, examen).status_code == 502
    usuario = db.query(Usuario).filter(Usuario.id == examen["usuario"]["id"]).first()
    assert not usuario.regeneraciones_usadas


def test_prompt_incluye_la_pregunta_anterior():
    prompt = generacion.construir_prompt_regeneracion("[Fragmento 1]\ntexto", PREGUNTA, "basico", "Indicación X")
    assert PREGUNTA["pregunta"] in prompt and "a) x" in prompt and "Indicación X" in prompt and "BÁSICO" in prompt
