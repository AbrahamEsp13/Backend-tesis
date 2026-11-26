"""Respuestas en curso: se guardan mientras responde y se recuperan al retomar."""
import datetime

from database import InicioIntento, ahora_utc


def iniciar(cliente, datos):
    respuesta = cliente.post(f"/api/cuestionarios/{datos['examen_id']}/inicio", headers=datos["estudiante"])
    assert respuesta.status_code == 200, respuesta.text
    return respuesta.json()


def guardar(cliente, datos, inicio_id, respuestas, seguridad=None, headers=None):
    return cliente.put(f"/api/inicios/{inicio_id}/borrador", headers=headers or datos["estudiante"],
                       json={"respuestas": respuestas, "seguridad": seguridad or {}})


def test_retomar_devuelve_lo_guardado(cliente, clase_con_examen, db):
    inicio = iniciar(cliente, clase_con_examen)
    assert inicio["borrador"] is None
    # Guarda por texto (se acepta por compatibilidad) y con seguridad de una pregunta sin responder
    assert guardar(cliente, clase_con_examen, inicio["inicio_id"], {"0": "a", "9": "b"}, {"0": True, "1": False}).status_code == 200

    guardado = db.query(InicioIntento).first()
    # Al iniciar, el examen recibió ids: el texto "a" se guarda como el id de esa opción; el índice 9 se descarta
    from database import Cuestionario
    id_a = db.query(Cuestionario).first().preguntas_json[0]["ids_opciones"][0]
    assert guardado.borrador_respuestas == {"0": id_a}
    assert guardado.borrador_seguridad == {"0": True}  # seguridad solo de lo respondido

    retomado = iniciar(cliente, clase_con_examen)
    assert retomado["reanudado"] and retomado["inicio_id"] == inicio["inicio_id"]
    assert retomado["borrador"]["respuestas"] == {"0": "a"}
    assert retomado["borrador"]["seguridad"] == {"0": True}


def test_borrador_con_ids_sigue_a_la_opcion_editada(cliente, clase_con_examen):
    # Una entrega previa asigna ids al examen
    cliente.post(f"/api/cuestionarios/{clase_con_examen['examen_id']}/intentos", headers=clase_con_examen["estudiante"],
                 json={"respuestas": {}})
    estudiante = next(e for e in cliente.get("/api/cuestionarios", headers=clase_con_examen["estudiante"]).json()["data"])
    id_b = estudiante["preguntas_json"][1]["ids_opciones"][1]
    inicio = iniciar(cliente, clase_con_examen)
    guardar(cliente, clase_con_examen, inicio["inicio_id"], {"1": id_b})

    docente = cliente.get("/api/cuestionarios", headers=clase_con_examen["docente"]).json()["data"][0]["preguntas_json"]
    docente[1]["opciones"] = ["a", "b editada"]
    docente[1]["respuesta_correcta"] = "b editada"
    cliente.put(f"/api/cuestionarios/{clase_con_examen['examen_id']}", headers=clase_con_examen["docente"], json={"preguntas_json": docente})

    assert iniciar(cliente, clase_con_examen)["borrador"]["respuestas"] == {"1": "b editada"}


def test_no_se_guarda_ajeno_entregado_ni_vencido(cliente, registrar, clase_con_examen, db):
    cliente.put(f"/api/cuestionarios/{clase_con_examen['examen_id']}/configuracion", headers=clase_con_examen["docente"],
                json={"tiempo_limite": 10})
    inicio_id = iniciar(cliente, clase_con_examen)["inicio_id"]

    otro, _ = registrar("beto", "estudiante")
    assert guardar(cliente, clase_con_examen, inicio_id, {"0": "a"}, headers=otro).status_code == 404

    # Se acabó el tiempo
    inicio = db.query(InicioIntento).first()
    inicio.fecha_inicio = ahora_utc() - datetime.timedelta(minutes=11)
    db.commit()
    assert guardar(cliente, clase_con_examen, inicio_id, {"0": "a"}).status_code == 409

    # Entregado
    nuevo_id = iniciar(cliente, clase_con_examen)["inicio_id"]
    cliente.post(f"/api/cuestionarios/{clase_con_examen['examen_id']}/intentos", headers=clase_con_examen["estudiante"],
                 json={"respuestas": {"0": "a"}, "inicio_id": nuevo_id})
    assert guardar(cliente, clase_con_examen, nuevo_id, {"0": "b"}).status_code == 409


def test_un_inicio_nuevo_empieza_sin_borrador(cliente, clase_con_examen):
    inicio_id = iniciar(cliente, clase_con_examen)["inicio_id"]
    guardar(cliente, clase_con_examen, inicio_id, {"0": "a"})
    cliente.post(f"/api/cuestionarios/{clase_con_examen['examen_id']}/intentos", headers=clase_con_examen["estudiante"],
                 json={"respuestas": {"0": "a"}, "inicio_id": inicio_id})
    nuevo = iniciar(cliente, clase_con_examen)
    assert not nuevo["reanudado"] and nuevo["borrador"] is None
