"""Id estable de pregunta: reportes, valoraciones, respuestas cortas y versiones siguen a su pregunta."""
import pytest

import generacion
import identificadores
import reportes
from database import Cuestionario, Reporte

CORTA = {"tipo": "respuesta_corta", "pregunta": "¿Qué define una clase?", "opciones": [],
         "respuesta_correcta": "Atributos y métodos", "justificacion_pedagogica": "j", "puntuacion": 2}


def preguntas_del_docente(cliente, datos):
    return next(e for e in cliente.get("/api/cuestionarios", headers=datos["docente"]).json()["data"]
                if e["id"] == datos["examen_id"])["preguntas_json"]


def guardar(cliente, datos, preguntas):
    respuesta = cliente.put(f"/api/cuestionarios/{datos['examen_id']}", headers=datos["docente"], json={"preguntas_json": preguntas})
    assert respuesta.status_code == 200, respuesta.text
    return respuesta.json()["preguntas_json"]


def entregar(cliente, datos, respuestas):
    respuesta = cliente.post(f"/api/cuestionarios/{datos['examen_id']}/intentos", headers=datos["estudiante"], json={"respuestas": respuestas})
    assert respuesta.status_code == 200, respuesta.text
    return respuesta.json()["intento"]


# ---------- Normalización ----------

def test_ids_de_pregunta_estables_y_unicos():
    preguntas = identificadores.normalizar_preguntas([{"pregunta": "a", "opciones": ["x"]}, {"pregunta": "b", "opciones": ["y"]}])
    ids = [p["id"] for p in preguntas]
    assert all(i.startswith("pr_") for i in ids) and len(set(ids)) == 2
    otra_vez = identificadores.normalizar_preguntas([preguntas[1], preguntas[0]])
    assert [p["id"] for p in otra_vez] == [ids[1], ids[0]]  # se conservan al reordenar
    repetidas = identificadores.normalizar_preguntas([preguntas[0], dict(preguntas[0])])
    assert repetidas[0]["id"] == ids[0] and repetidas[1]["id"] != ids[0]
    assert identificadores.indice_de_pregunta(preguntas, ids[1]) == 1
    assert identificadores.indice_de_pregunta(preguntas, "0") == 0
    assert identificadores.indice_de_pregunta(preguntas, "pr_nada") is None


# ---------- Reportes ----------

def reportar(cliente, datos, intento_id, pregunta, headers=None):
    return cliente.post(f"/api/intentos/{intento_id}/reportes", headers=headers or datos["estudiante"],
                        json={"pregunta": pregunta, "motivo": "ambigua"})


def reportes_del_docente(cliente, datos):
    return cliente.get(f"/api/cuestionarios/{datos['examen_id']}/intentos", headers=datos["docente"]).json()["reportes"]


def test_reportes_siguen_a_su_pregunta(cliente, clase_con_examen):
    intento = entregar(cliente, clase_con_examen, {"0": "a", "1": "a"})
    assert reportar(cliente, clase_con_examen, intento["id"], 0).status_code == 200
    assert reportar(cliente, clase_con_examen, intento["id"], 1).status_code == 200

    preguntas = preguntas_del_docente(cliente, clase_con_examen)
    guardar(cliente, clase_con_examen, [preguntas[1], preguntas[0]])  # intercambiarlas no choca la restricción única
    assert {r["texto_pregunta"]: r["pregunta"] for r in reportes_del_docente(cliente, clase_con_examen)} == {"p1": 1, "p2": 0}

    guardar(cliente, clase_con_examen, [preguntas[0]])  # se borra la "p2": su reporte desaparece
    assert [(r["texto_pregunta"], r["pregunta"]) for r in reportes_del_docente(cliente, clase_con_examen)] == [("p1", 0)]


def test_regenerar_una_pregunta_descarta_sus_reportes(cliente, clase_con_examen):
    intento = entregar(cliente, clase_con_examen, {"0": "a"})
    reportar(cliente, clase_con_examen, intento["id"], 0)
    preguntas = preguntas_del_docente(cliente, clase_con_examen)
    nueva = {k: v for k, v in preguntas[0].items() if k not in ("id", "ids_opciones", "id_correcta")}  # como la propuesta de la IA
    guardar(cliente, clase_con_examen, [{**nueva, "pregunta": "otra"}, preguntas[1]])
    assert reportes_del_docente(cliente, clase_con_examen) == []


def test_migracion_completa_ids_de_reportes(db, registrar, crear_cuestionario):
    _, docente = registrar("docente", "docente")
    examen_id = crear_cuestionario(docente["id"], preguntas=identificadores.normalizar_preguntas(
        [{"pregunta": "p1", "opciones": ["a"]}, {"pregunta": "p2", "opciones": ["a"]}]))
    db.add_all([
        Reporte(cuestionario_id=examen_id, estudiante_id=9, intento_id=1, pregunta=1, texto_pregunta="p2", motivo="otro"),
        Reporte(cuestionario_id=examen_id, estudiante_id=8, intento_id=2, pregunta=0, texto_pregunta="texto viejo", motivo="otro"),
    ])
    db.commit()
    assert reportes.completar_ids(db) == 1
    preguntas = db.query(Cuestionario).first().preguntas_json
    assert {r.texto_pregunta: r.pregunta_id for r in db.query(Reporte).all()} == {"p2": preguntas[1]["id"], "texto viejo": None}


# ---------- Valoraciones ----------

def test_valorar_por_id(cliente, clase_con_examen, db):
    identificadores.migrar(db)  # como al arrancar: los exámenes existentes reciben sus ids
    preguntas = preguntas_del_docente(cliente, clase_con_examen)
    ruta = f"/api/cuestionarios/{clase_con_examen['examen_id']}/preguntas/{{}}/valoracion"
    assert cliente.put(ruta.format(preguntas[1]["id"]), headers=clase_con_examen["docente"], json={"util": True}).status_code == 200
    assert preguntas_del_docente(cliente, clase_con_examen)[1]["valoracion"]["util"] is True
    assert cliente.put(ruta.format("0"), headers=clase_con_examen["docente"], json={"util": False, "motivo": "ambigua"}).status_code == 200
    assert cliente.put(ruta.format("pr_inexistente"), headers=clase_con_examen["docente"], json={"util": True}).status_code == 404


# ---------- Respuestas cortas ----------

@pytest.fixture
def examen_con_corta(clase_con_examen, db, monkeypatch):
    examen = db.query(Cuestionario).first()
    examen.preguntas_json = [examen.preguntas_json[0], CORTA]
    db.commit()
    monkeypatch.setattr(generacion, "calificar_respuestas_cortas",
                        lambda items: {i["numero"]: {"puntaje": 0.5, "comentario": "Te faltó algo."} for i in items})
    return clase_con_examen


def test_respuesta_corta_sigue_a_su_pregunta(cliente, examen_con_corta):
    intento = entregar(cliente, examen_con_corta, {"0": "a", "1": "Atributos"})
    preguntas = preguntas_del_docente(cliente, examen_con_corta)
    guardar(cliente, examen_con_corta, [preguntas[1]])  # se borra la primera: la corta pasa a la posición 0

    revision = cliente.get(f"/api/intentos/{intento['id']}", headers=examen_con_corta["docente"]).json()["intento"]
    assert revision["respuestas"] == {"0": "Atributos"}
    assert revision["respuestas_cortas"]["0"]["texto"] == "Atributos" and "1" not in revision["respuestas_cortas"]

    corregido = cliente.put(f"/api/intentos/{intento['id']}/respuestas-cortas/0", headers=examen_con_corta["docente"], json={"puntaje": 1}).json()
    assert corregido["intento"]["puntos_obtenidos"] == 2 and corregido["intento"]["puntos_totales"] == 2
    assert corregido["intento"]["respuestas"] == {"0": "Atributos"}


# ---------- Versiones ----------

def test_orden_sigue_a_las_preguntas(cliente, clase_con_examen):
    cliente.put(f"/api/cuestionarios/{clase_con_examen['examen_id']}/configuracion", headers=clase_con_examen["docente"],
                json={"barajar_preguntas": True, "barajar_opciones": True})
    inicio = cliente.post(f"/api/cuestionarios/{clase_con_examen['examen_id']}/inicio", headers=clase_con_examen["estudiante"]).json()
    vistos = [p["pregunta"] for p in inicio["preguntas_json"]]
    intento = cliente.post(f"/api/cuestionarios/{clase_con_examen['examen_id']}/intentos", headers=clase_con_examen["estudiante"],
                           json={"respuestas": {}, "inicio_id": inicio["inicio_id"]}).json()["intento"]

    preguntas = preguntas_del_docente(cliente, clase_con_examen)
    actuales = guardar(cliente, clase_con_examen, [preguntas[1], preguntas[0]])  # el docente las intercambia
    orden = cliente.get(f"/api/intentos/{intento['id']}", headers=clase_con_examen["docente"]).json()["intento"]["orden"]
    assert [actuales[i]["pregunta"] for i in orden["preguntas"]] == vistos  # el estudiante las vio en este orden
    assert set(orden["opciones"]) == {"0", "1"}
