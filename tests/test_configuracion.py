"""4.1 Configuración del examen: fecha límite, tiempo límite, intentos máximos y respuestas visibles."""
import datetime

import pytest

import configuracion
import practicas
from database import Cuestionario, InicioIntento, ahora_utc


def configurar(cliente, datos, **cambios):
    return cliente.put(f"/api/cuestionarios/{datos['examen_id']}/configuracion", headers=datos["docente"], json=cambios)


def iniciar(cliente, datos):
    return cliente.post(f"/api/cuestionarios/{datos['examen_id']}/inicio", headers=datos["estudiante"])


def entregar(cliente, datos, inicio_id=None, respuestas=None):
    cuerpo = {"respuestas": respuestas if respuestas is not None else {"0": "a", "1": "a"}}
    if inicio_id is not None:
        cuerpo["inicio_id"] = inicio_id
    return cliente.post(f"/api/cuestionarios/{datos['examen_id']}/intentos", headers=datos["estudiante"], json=cuerpo)


def mi_estado(cliente, datos):
    lista = cliente.get("/api/cuestionarios", headers=datos["estudiante"]).json()["data"]
    return next(e for e in lista if e["id"] == datos["examen_id"])["mi_estado"]


def mover_inicio(db, inicio_id, minutos):
    """Simula que el estudiante inició hace `minutos`."""
    inicio = db.query(InicioIntento).filter(InicioIntento.id == inicio_id).first()
    inicio.fecha_inicio = ahora_utc() - datetime.timedelta(minutes=minutos)
    db.commit()


# ---------- Configurar ----------

def test_configurar_y_validaciones(cliente, clase_con_examen):
    respuesta = configurar(cliente, clase_con_examen, fecha_limite="2030-01-01T12:00:00-06:00", tiempo_limite=30,
                           intentos_maximos=2, mostrar_respuestas="despues_del_cierre")
    assert respuesta.status_code == 200, respuesta.text
    config = respuesta.json()["configuracion"]
    assert config["fecha_limite"].startswith("2030-01-01T18:00:00")  # guardada en UTC
    assert config["tiempo_limite"] == 30 and config["intentos_maximos"] == 2

    assert configurar(cliente, clase_con_examen, mostrar_respuestas="despues_del_cierre").status_code == 400
    assert configurar(cliente, clase_con_examen, tiempo_limite=0).status_code == 422
    assert cliente.put(f"/api/cuestionarios/{clase_con_examen['examen_id']}/configuracion",
                       headers=clase_con_examen["estudiante"], json={}).status_code == 403


def test_sin_configuracion_todo_sigue_igual(cliente, clase_con_examen):
    assert entregar(cliente, clase_con_examen).status_code == 200  # sin inicio
    estado = mi_estado(cliente, clase_con_examen)
    assert estado["puede_iniciar"] and estado["intentos_restantes"] is None and estado["respuestas_visibles"]


# ---------- Inicio y tiempo ----------

def test_inicio_se_reutiliza_al_recargar(cliente, clase_con_examen):
    configurar(cliente, clase_con_examen, tiempo_limite=20)
    primero = iniciar(cliente, clase_con_examen).json()
    segundo = iniciar(cliente, clase_con_examen).json()
    assert segundo["inicio_id"] == primero["inicio_id"] and segundo["reanudado"]
    assert 1190 <= segundo["segundos_restantes"] <= 1200


def test_tiempo_limite_exige_inicio_y_marca_fuera_de_tiempo(cliente, clase_con_examen, db):
    configurar(cliente, clase_con_examen, tiempo_limite=10)
    assert entregar(cliente, clase_con_examen).status_code == 400  # sin inicio

    a_tiempo = iniciar(cliente, clase_con_examen).json()["inicio_id"]
    mover_inicio(db, a_tiempo, 5)
    intento = entregar(cliente, clase_con_examen, a_tiempo).json()["intento"]
    assert intento["fuera_de_tiempo"] is False
    assert 295 <= intento["tiempo_segundos"] <= 305  # medido en el servidor
    assert entregar(cliente, clase_con_examen, a_tiempo).status_code == 409  # no se entrega dos veces

    tarde = iniciar(cliente, clase_con_examen).json()["inicio_id"]
    mover_inicio(db, tarde, 12)
    assert entregar(cliente, clase_con_examen, tarde).json()["intento"]["fuera_de_tiempo"] is True
    docente = cliente.get(f"/api/cuestionarios/{clase_con_examen['examen_id']}/intentos", headers=clase_con_examen["docente"]).json()
    assert sorted(i["fuera_de_tiempo"] for i in docente["data"]) == [False, True]


def test_inicio_ajeno_no_sirve(cliente, registrar, clase_con_examen):
    inicio_id = iniciar(cliente, clase_con_examen).json()["inicio_id"]
    otro_examen = {**clase_con_examen}
    otro, _ = registrar("beto", "estudiante")
    cliente.post("/api/clases/unirse", headers=otro, json={"codigo": clase_con_examen["clase"]["codigo"]})
    otro_examen["estudiante"] = otro
    assert entregar(cliente, otro_examen, inicio_id).status_code == 400


# ---------- Intentos máximos y fecha límite ----------

def test_intentos_maximos_cuentan_al_iniciar(cliente, clase_con_examen):
    entregar(cliente, clase_con_examen)  # intento anterior sin inicio: también cuenta
    configurar(cliente, clase_con_examen, intentos_maximos=2)
    inicio_id = iniciar(cliente, clase_con_examen).json()["inicio_id"]
    assert mi_estado(cliente, clase_con_examen)["intentos_restantes"] == 0
    # Con un inicio en curso todavía puede retomar y entregar
    assert mi_estado(cliente, clase_con_examen)["puede_iniciar"] is True
    assert iniciar(cliente, clase_con_examen).json()["inicio_id"] == inicio_id
    assert entregar(cliente, clase_con_examen, inicio_id).status_code == 200

    respuesta = iniciar(cliente, clase_con_examen)
    assert respuesta.status_code == 403 and "intentos" in respuesta.json()["detail"]
    assert mi_estado(cliente, clase_con_examen)["motivo"] == "sin_intentos"


def test_fecha_limite_cierra_el_examen(cliente, clase_con_examen):
    pasada = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)).isoformat()
    configurar(cliente, clase_con_examen, fecha_limite=pasada)
    assert iniciar(cliente, clase_con_examen).status_code == 403
    assert mi_estado(cliente, clase_con_examen)["motivo"] == "cerrado"
    # Sin tiempo ni intentos limitados puede entregar sin inicio, pero queda marcado
    assert entregar(cliente, clase_con_examen).json()["intento"]["fuera_de_tiempo"] is True


def test_segundos_restantes_respeta_la_fecha_limite(cliente, clase_con_examen):
    en_5_min = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)).isoformat()
    configurar(cliente, clase_con_examen, fecha_limite=en_5_min, tiempo_limite=60)
    assert 290 <= iniciar(cliente, clase_con_examen).json()["segundos_restantes"] <= 300


# ---------- Respuestas visibles ----------

@pytest.fixture
def sin_ia(monkeypatch):
    monkeypatch.setattr(practicas, "generar_practica", lambda contexto, n, errores: [
        {"pregunta": "n", "opciones": ["a) x", "b) y"], "respuesta_correcta": "a) x", "justificacion_pedagogica": "j",
         "puntuacion": 1, "fragmento": 1, "cita": ""}])


def test_nunca_mostrar_respuestas(cliente, clase_con_examen, sin_ia):
    configurar(cliente, clase_con_examen, mostrar_respuestas="nunca")
    resultado = entregar(cliente, clase_con_examen).json()
    assert resultado["respuestas_visibles"] is False
    assert resultado["respuestas_correctas"] is None and resultado["justificaciones"] is None
    assert resultado["intento"]["calificacion"] == 2.5  # la calificación sí se ve

    intento_id = resultado["intento"]["id"]
    revision = cliente.get(f"/api/intentos/{intento_id}", headers=clase_con_examen["estudiante"]).json()
    assert revision["respuestas_visibles"] is False
    assert all("respuesta_correcta" not in p and "justificacion_pedagogica" not in p for p in revision["examen"]["preguntas_json"])
    # El docente sí ve la clave
    del_docente = cliente.get(f"/api/intentos/{intento_id}", headers=clase_con_examen["docente"]).json()
    assert del_docente["examen"]["preguntas_json"][0]["respuesta_correcta"] == "a"
    # Y no hay práctica (revelaría qué falló y la clave)
    assert cliente.post(f"/api/intentos/{intento_id}/practica", headers=clase_con_examen["estudiante"]).status_code == 403


def test_respuestas_despues_del_cierre(cliente, clase_con_examen, db):
    en_1_hora = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)).isoformat()
    configurar(cliente, clase_con_examen, fecha_limite=en_1_hora, mostrar_respuestas="despues_del_cierre")
    intento_id = entregar(cliente, clase_con_examen).json()["intento"]["id"]
    ruta = f"/api/intentos/{intento_id}"
    assert cliente.get(ruta, headers=clase_con_examen["estudiante"]).json()["respuestas_visibles"] is False

    examen = db.query(Cuestionario).first()
    examen.fecha_limite = ahora_utc() - datetime.timedelta(minutes=1)
    db.commit()
    revision = cliente.get(ruta, headers=clase_con_examen["estudiante"]).json()
    assert revision["respuestas_visibles"] is True
    assert revision["examen"]["preguntas_json"][0]["respuesta_correcta"] == "a"


def test_borrar_el_examen_borra_los_inicios(cliente, clase_con_examen, db):
    iniciar(cliente, clase_con_examen)
    assert db.query(InicioIntento).count() == 1
    cliente.delete(f"/api/cuestionarios/{clase_con_examen['examen_id']}", headers=clase_con_examen["docente"])
    db.expire_all()
    assert db.query(InicioIntento).count() == 0


def test_evaluar_entrega_con_gracia():
    examen = Cuestionario(tiempo_limite=10)
    inicio = InicioIntento(fecha_inicio=datetime.datetime(2026, 1, 1, 12, 0, 0))
    justo = datetime.datetime(2026, 1, 1, 12, 10, 0) + datetime.timedelta(seconds=configuracion.GRACIA_SEGUNDOS)
    assert configuracion.evaluar_entrega(examen, inicio, justo) == (600 + configuracion.GRACIA_SEGUNDOS, False)
    assert configuracion.evaluar_entrega(examen, inicio, justo + datetime.timedelta(seconds=1))[1] is True
