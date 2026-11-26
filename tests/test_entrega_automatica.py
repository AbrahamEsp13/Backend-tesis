"""Entrega automática: un intento vencido con respuestas guardadas se entrega solo."""
import datetime

import configuracion
from database import InicioIntento, Intento, ahora_utc


def preparar(cliente, datos, db, respuestas, minutos_atras, **config):
    """Configura el examen, inicia, guarda un borrador y mueve el inicio `minutos_atras` al pasado."""
    if config:
        cliente.put(f"/api/cuestionarios/{datos['examen_id']}/configuracion", headers=datos["docente"], json=config)
    inicio_id = cliente.post(f"/api/cuestionarios/{datos['examen_id']}/inicio", headers=datos["estudiante"]).json()["inicio_id"]
    if respuestas is not None:
        assert cliente.put(f"/api/inicios/{inicio_id}/borrador", headers=datos["estudiante"],
                           json={"respuestas": respuestas, "seguridad": {"0": True}}).status_code == 200
    inicio = db.query(InicioIntento).filter(InicioIntento.id == inicio_id).first()
    inicio.fecha_inicio = ahora_utc() - datetime.timedelta(minutes=minutos_atras)
    db.commit()
    return inicio_id


def intentos_del_docente(cliente, datos):
    return cliente.get(f"/api/cuestionarios/{datos['examen_id']}/intentos", headers=datos["docente"]).json()["data"]


def test_vencido_con_borrador_se_entrega_al_consultar(cliente, clase_con_examen, db):
    inicio_id = preparar(cliente, clase_con_examen, db, {"0": "a", "1": "a"}, minutos_atras=12, tiempo_limite=10)
    intentos = intentos_del_docente(cliente, clase_con_examen)
    assert len(intentos) == 1
    intento = intentos[0]
    assert intento["entrega_automatica"] is True and intento["fuera_de_tiempo"] is False
    assert intento["puntos_obtenidos"] == 1 and intento["puntos_totales"] == 4
    assert intento["respuestas"] == {"0": "a", "1": "a"}
    assert intento["seguridad"] == {"0": True}
    assert intento["tiempo_segundos"] == 600  # el tiempo límite completo
    db.expire_all()
    assert db.query(InicioIntento).filter(InicioIntento.id == inicio_id).first().entregado is True

    # No se entrega dos veces, y el navegador ya no puede entregarlo
    assert len(intentos_del_docente(cliente, clase_con_examen)) == 1
    tarde = cliente.post(f"/api/cuestionarios/{clase_con_examen['examen_id']}/intentos", headers=clase_con_examen["estudiante"],
                         json={"respuestas": {"0": "a"}, "inicio_id": inicio_id})
    assert tarde.status_code == 409


def test_el_estudiante_tambien_lo_ve(cliente, clase_con_examen, db):
    preparar(cliente, clase_con_examen, db, {"0": "a"}, minutos_atras=12, tiempo_limite=10)
    mios = cliente.get("/api/intentos/mios", headers=clase_con_examen["estudiante"]).json()["data"]
    assert len(mios) == 1 and mios[0]["entrega_automatica"] is True


def test_dentro_del_tiempo_o_de_la_gracia_no_se_entrega(cliente, clase_con_examen, db):
    preparar(cliente, clase_con_examen, db, {"0": "a"}, minutos_atras=10.5, tiempo_limite=10)  # 30 s de 60 de gracia
    assert intentos_del_docente(cliente, clase_con_examen) == []


def test_sin_borrador_o_sin_limite_no_se_entrega(cliente, clase_con_examen, db):
    preparar(cliente, clase_con_examen, db, None, minutos_atras=30, tiempo_limite=10)  # sin respuestas: abandono
    assert intentos_del_docente(cliente, clase_con_examen) == []
    assert configuracion.entregar_vencidos(db) == 0


def test_sin_limite_un_inicio_abandonado_no_se_entrega(cliente, clase_con_examen, db):
    preparar(cliente, clase_con_examen, db, {"0": "a"}, minutos_atras=60 * 24)
    assert intentos_del_docente(cliente, clase_con_examen) == []


def test_por_fecha_limite(cliente, clase_con_examen, db):
    en_1_min = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=1)).isoformat()
    preparar(cliente, clase_con_examen, db, {"0": "a"}, minutos_atras=0, fecha_limite=en_1_min)
    from database import Cuestionario
    examen = db.query(Cuestionario).first()
    examen.fecha_limite = ahora_utc() - datetime.timedelta(minutes=5)
    db.commit()
    intento = db.query(Intento).count()
    assert intento == 0
    assert configuracion.entregar_vencidos(db, estudiante_id=clase_con_examen["usuario_estudiante"]["id"]) == 1
    guardado = db.query(Intento).first()
    assert guardado.entrega_automatica and guardado.fecha == examen.fecha_limite


def test_inicio_docente_muestra_la_entrega(cliente, clase_con_examen, db):
    preparar(cliente, clase_con_examen, db, {"0": "a"}, minutos_atras=12, tiempo_limite=10)
    resumen = cliente.get("/api/docente/resumen", headers=clase_con_examen["docente"]).json()
    assert len(resumen["entregas_recientes"]) == 1
