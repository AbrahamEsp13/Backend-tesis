"""4.6 Clave por identificador estable: editar opciones o borrar preguntas no descuadra los intentos."""
import identificadores
from database import Cuestionario, Intento

PREGUNTAS = [
    {"pregunta": "p1", "opciones": ["a", "b"], "respuesta_correcta": "a", "justificacion_pedagogica": "j1", "puntuacion": 1},
    {"pregunta": "p2", "opciones": ["a", "b"], "respuesta_correcta": "b", "justificacion_pedagogica": "j2", "puntuacion": 3},
]


# ---------- Normalización ----------

def test_normalizar_asigna_ids_y_la_correcta():
    preguntas = identificadores.normalizar_preguntas(PREGUNTAS)
    todos = [i for p in preguntas for i in p["ids_opciones"]]
    assert len(todos) == 4 and len(set(todos)) == 4
    assert all(i.startswith(identificadores.PREFIJO) for i in todos)
    assert preguntas[0]["id_correcta"] == preguntas[0]["ids_opciones"][0]
    assert preguntas[1]["id_correcta"] == preguntas[1]["ids_opciones"][1]
    assert "ids_opciones" not in PREGUNTAS[0]  # no modifica la lista original


def test_editar_texto_conserva_el_id():
    preguntas = identificadores.normalizar_preguntas(PREGUNTAS)
    editada = [dict(p) for p in preguntas]
    editada[0] = {**editada[0], "opciones": ["a corregida", "b"], "respuesta_correcta": "a corregida"}
    otra_vez = identificadores.normalizar_preguntas(editada)
    assert otra_vez[0]["ids_opciones"] == preguntas[0]["ids_opciones"]
    assert otra_vez[0]["id_correcta"] == preguntas[0]["id_correcta"]


def test_id_correcta_del_cliente_se_ignora_y_los_repetidos_se_cambian():
    preguntas = identificadores.normalizar_preguntas(PREGUNTAS)
    alterada = [dict(p) for p in preguntas]
    alterada[0]["id_correcta"] = alterada[0]["ids_opciones"][1]
    alterada[1]["ids_opciones"] = list(alterada[0]["ids_opciones"])  # ids repetidos entre preguntas
    resultado = identificadores.normalizar_preguntas(alterada)
    assert resultado[0]["id_correcta"] == resultado[0]["ids_opciones"][0]  # sale de respuesta_correcta
    assert not set(resultado[0]["ids_opciones"]) & set(resultado[1]["ids_opciones"])


def test_respuestas_a_ids_acepta_ids_y_textos():
    preguntas = identificadores.normalizar_preguntas(PREGUNTAS)
    ids0, ids1 = preguntas[0]["ids_opciones"], preguntas[1]["ids_opciones"]
    convertidas = identificadores.respuestas_a_ids(preguntas, {"0": ids0[1], "1": "b", "5": "a"})
    assert convertidas == {"0": ids0[1], "1": ids1[1]}


# ---------- Endpoints ----------

def entregar(cliente, datos, respuestas, seguridad=None):
    respuesta = cliente.post(f"/api/cuestionarios/{datos['examen_id']}/intentos", headers=datos["estudiante"],
                             json={"respuestas": respuestas, "seguridad": seguridad or {}})
    assert respuesta.status_code == 200, respuesta.text
    return respuesta.json()


def examen_del_estudiante(cliente, datos):
    lista = cliente.get("/api/cuestionarios", headers=datos["estudiante"]).json()["data"]
    return next(e for e in lista if e["id"] == datos["examen_id"])


def guardar_preguntas(cliente, datos, preguntas):
    respuesta = cliente.put(f"/api/cuestionarios/{datos['examen_id']}", headers=datos["docente"], json={"preguntas_json": preguntas})
    assert respuesta.status_code == 200, respuesta.text
    return respuesta.json()["preguntas_json"]


def preguntas_del_docente(cliente, datos):
    lista = cliente.get("/api/cuestionarios", headers=datos["docente"]).json()["data"]
    return next(e for e in lista if e["id"] == datos["examen_id"])["preguntas_json"]


def test_estudiante_recibe_ids_pero_no_la_clave(cliente, clase_con_examen):
    entregar(cliente, clase_con_examen, {"0": "a"})  # la primera entrega asigna ids al examen antiguo
    pregunta = examen_del_estudiante(cliente, clase_con_examen)["preguntas_json"][0]
    assert len(pregunta["ids_opciones"]) == 2
    assert "id_correcta" not in pregunta and "respuesta_correcta" not in pregunta


def test_entregar_con_ids_guarda_ids_y_devuelve_textos(cliente, clase_con_examen, db):
    entregar(cliente, clase_con_examen, {})  # asigna ids
    preguntas = examen_del_estudiante(cliente, clase_con_examen)["preguntas_json"]
    elegidas = {"0": preguntas[0]["ids_opciones"][0], "1": preguntas[1]["ids_opciones"][0]}
    resultado = entregar(cliente, clase_con_examen, elegidas)
    assert resultado["intento"]["puntos_obtenidos"] == 1  # acierta la 1 (1 pt), falla la 2
    assert resultado["intento"]["respuestas"] == {"0": "a", "1": "a"}
    guardado = db.query(Intento).filter(Intento.id == resultado["intento"]["id"]).first()
    assert guardado.respuestas == elegidas


def test_editar_una_opcion_no_descuadra_el_intento(cliente, clase_con_examen):
    intento = entregar(cliente, clase_con_examen, {"0": "a", "1": "b"})["intento"]
    preguntas = preguntas_del_docente(cliente, clase_con_examen)
    preguntas[0]["opciones"] = ["a (corregida)", "b"]
    preguntas[0]["respuesta_correcta"] = "a (corregida)"
    guardar_preguntas(cliente, clase_con_examen, preguntas)

    revision = cliente.get(f"/api/intentos/{intento['id']}", headers=clase_con_examen["estudiante"]).json()
    assert revision["intento"]["respuestas"] == {"0": "a (corregida)", "1": "b"}
    assert revision["examen"]["preguntas_json"][0]["respuesta_correcta"] == "a (corregida)"
    docente = cliente.get(f"/api/cuestionarios/{clase_con_examen['examen_id']}/intentos", headers=clase_con_examen["docente"]).json()
    assert docente["data"][0]["respuestas"] == {"0": "a (corregida)", "1": "b"}


def test_borrar_una_pregunta_mueve_las_respuestas(cliente, clase_con_examen):
    intento = entregar(cliente, clase_con_examen, {"0": "a", "1": "a"}, seguridad={"0": True, "1": False})["intento"]
    preguntas = preguntas_del_docente(cliente, clase_con_examen)
    guardar_preguntas(cliente, clase_con_examen, preguntas[1:])  # se borra la pregunta 1

    revision = cliente.get(f"/api/intentos/{intento['id']}", headers=clase_con_examen["estudiante"]).json()
    # La respuesta a la antigua pregunta 2 ahora está en el índice 0; la de la borrada desaparece
    assert revision["intento"]["respuestas"] == {"0": "a"}
    assert revision["intento"]["seguridad"] == {"0": False}


def test_guardar_devuelve_ids_estables(cliente, clase_con_examen):
    preguntas = preguntas_del_docente(cliente, clase_con_examen)
    primera = guardar_preguntas(cliente, clase_con_examen, preguntas)
    segunda = guardar_preguntas(cliente, clase_con_examen, primera)
    assert [p["ids_opciones"] for p in primera] == [p["ids_opciones"] for p in segunda]


def test_la_practica_usa_las_respuestas_traducidas(cliente, clase_con_examen, monkeypatch):
    import practicas
    llamadas = []
    monkeypatch.setattr(practicas, "generar_practica", lambda contexto, n, errores: llamadas.append(errores) or [
        {"pregunta": "n", "opciones": ["a) x", "b) y"], "respuesta_correcta": "a) x", "justificacion_pedagogica": "j",
         "puntuacion": 1, "fragmento": 1, "cita": ""}])
    intento = entregar(cliente, clase_con_examen, {"0": "a", "1": "a"})["intento"]
    preguntas = preguntas_del_docente(cliente, clase_con_examen)
    preguntas[1]["opciones"] = ["a editada", "b"]
    guardar_preguntas(cliente, clase_con_examen, preguntas)
    assert cliente.post(f"/api/intentos/{intento['id']}/practica", headers=clase_con_examen["estudiante"]).status_code == 200
    assert llamadas[0] == [{"pregunta": "p2", "respuesta_correcta": "b", "elegida": "a editada"}]


# ---------- Migración ----------

def test_migrar_convierte_examenes_e_intentos_antiguos(db, registrar, crear_cuestionario):
    _, docente = registrar("docente", "docente")
    examen_id = crear_cuestionario(docente["id"])
    db.add(Intento(cuestionario_id=examen_id, estudiante_id=99, respuestas={"0": "a", "1": "a", "7": "x"},
                   puntos_obtenidos=1, puntos_totales=4, calificacion=2.5))
    db.commit()

    assert identificadores.migrar(db) == {"cuestionarios": 1, "intentos": 1}
    db.expire_all()
    preguntas = db.query(Cuestionario).first().preguntas_json
    intento = db.query(Intento).first()
    assert intento.respuestas == {"0": preguntas[0]["ids_opciones"][0], "1": preguntas[1]["ids_opciones"][0], "7": "x"}
    # Idempotente
    assert identificadores.migrar(db) == {"cuestionarios": 0, "intentos": 0}
