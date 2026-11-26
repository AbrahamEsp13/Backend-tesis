"""2.6 Versiones del examen: orden de preguntas y opciones por intento."""
import random

import versiones
from database import Cuestionario

PREGUNTAS = [
    {"pregunta": f"p{i}", "opciones": ["a) x", "b) y", "c) z"], "respuesta_correcta": "a) x",
     "ids_opciones": [f"op_{i}a", f"op_{i}b", f"op_{i}c"], "id_correcta": f"op_{i}a", "justificacion_pedagogica": "j"}
    for i in range(6)
]


# ---------- Lógica ----------

def test_sin_barajar_no_hay_orden():
    assert versiones.generar_orden(PREGUNTAS, False, False) is None


def test_generar_orden_es_una_permutacion():
    orden = versiones.generar_orden(PREGUNTAS, True, True, random.Random(1))
    assert sorted(orden["preguntas"]) == list(range(6))
    assert all(sorted(v) == [0, 1, 2] for v in orden["opciones"].values()) and len(orden["opciones"]) == 6
    solo_preguntas = versiones.generar_orden(PREGUNTAS, True, False, random.Random(1))
    assert solo_preguntas["opciones"] == {}


def test_preguntas_en_orden_sin_clave_y_con_ids_reordenados():
    orden = {"preguntas": [2, 0, 1, 3, 4, 5], "opciones": {"2": [2, 0, 1]}}
    vista = versiones.preguntas_en_orden(PREGUNTAS, orden)
    assert [p["indice_original"] for p in vista] == [2, 0, 1, 3, 4, 5]
    primera = vista[0]
    assert primera["opciones"] == ["c) z", "a) x", "b) y"] and primera["ids_opciones"] == ["op_2c", "op_2a", "op_2b"]
    assert primera["opciones_barajadas"] is True and "opciones_barajadas" not in vista[1]
    assert all("respuesta_correcta" not in p and "id_correcta" not in p for p in vista)


def test_normalizar_tras_editar_el_examen():
    orden = {"preguntas": [5, 9, 1, 1, 0], "opciones": {"1": [1, 0], "0": [2, 1, 0], "9": [0]}}
    normal = versiones.normalizar_orden(orden, PREGUNTAS)
    assert normal["preguntas"] == [5, 1, 0, 2, 3, 4]  # sin el 9 ni el repetido; las demás al final
    assert normal["opciones"] == {"0": [2, 1, 0]}  # la 1 tenía otro número de opciones
    assert versiones.normalizar_orden(None, PREGUNTAS[:2]) == {"preguntas": [0, 1], "opciones": {}}


# ---------- Flujo completo ----------

def configurar(cliente, datos, **cambios):
    respuesta = cliente.put(f"/api/cuestionarios/{datos['examen_id']}/configuracion", headers=datos["docente"], json=cambios)
    assert respuesta.status_code == 200, respuesta.text
    return respuesta.json()["configuracion"]


def iniciar(cliente, datos):
    return cliente.post(f"/api/cuestionarios/{datos['examen_id']}/inicio", headers=datos["estudiante"]).json()


def usar_examen_largo(db):
    examen = db.query(Cuestionario).first()
    examen.preguntas_json = [{k: v for k, v in p.items() if k not in ("ids_opciones", "id_correcta")} for p in PREGUNTAS]
    db.commit()


def test_sin_barajar_el_inicio_devuelve_el_orden_original(cliente, clase_con_examen):
    inicio = iniciar(cliente, clase_con_examen)
    assert [p["indice_original"] for p in inicio["preguntas_json"]] == [0, 1]
    assert all("respuesta_correcta" not in p and len(p["ids_opciones"]) == 2 for p in inicio["preguntas_json"])


def test_barajar_entregar_y_revisar(cliente, clase_con_examen, db):
    usar_examen_largo(db)
    config = configurar(cliente, clase_con_examen, barajar_preguntas=True, barajar_opciones=True)
    assert config["barajar_preguntas"] and config["barajar_opciones"]

    inicio = iniciar(cliente, clase_con_examen)
    vista = inicio["preguntas_json"]
    assert sorted(p["indice_original"] for p in vista) == list(range(6))
    assert iniciar(cliente, clase_con_examen)["preguntas_json"] == vista  # al retomar, el mismo orden

    # Responde todo bien usando el id de "a) x" en cada pregunta, con la clave del índice original
    respuestas = {str(p["indice_original"]): p["ids_opciones"][p["opciones"].index("a) x")] for p in vista}
    resultado = cliente.post(f"/api/cuestionarios/{clase_con_examen['examen_id']}/intentos", headers=clase_con_examen["estudiante"],
                             json={"respuestas": respuestas, "inicio_id": inicio["inicio_id"]}).json()
    assert resultado["intento"]["calificacion"] == 10
    orden = resultado["intento"]["orden"]
    assert orden["preguntas"] == [p["indice_original"] for p in vista]
    assert len(orden["opciones"]) == 6

    revision = cliente.get(f"/api/intentos/{resultado['intento']['id']}", headers=clase_con_examen["estudiante"]).json()
    assert revision["intento"]["orden"] == orden and revision["intento"]["respuestas"]["0"] == "a) x"


def test_cada_intento_puede_tener_otro_orden(cliente, clase_con_examen, db):
    usar_examen_largo(db)
    configurar(cliente, clase_con_examen, barajar_preguntas=True)
    ordenes = set()
    for _ in range(6):
        inicio = iniciar(cliente, clase_con_examen)
        ordenes.add(tuple(p["indice_original"] for p in inicio["preguntas_json"]))
        cliente.post(f"/api/cuestionarios/{clase_con_examen['examen_id']}/intentos", headers=clase_con_examen["estudiante"],
                     json={"respuestas": {}, "inicio_id": inicio["inicio_id"]})
    assert len(ordenes) > 1  # 720 órdenes posibles: que salgan 6 iguales es prácticamente imposible
