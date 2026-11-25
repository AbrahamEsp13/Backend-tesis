"""Entrega de intentos por estudiantes y resultados para el docente."""


def entregar(cliente, encabezados, examen_id, respuestas, tiempo=None):
    return cliente.post(f"/api/cuestionarios/{examen_id}/intentos", headers=encabezados,
                        json={"respuestas": respuestas, "tiempo_segundos": tiempo})


def test_estudiante_recibe_el_examen_sin_la_clave(cliente, clase_con_examen):
    d = clase_con_examen
    lista = cliente.get("/api/cuestionarios", headers=d["estudiante"]).json()["data"]
    assert len(lista) == 1
    assert lista[0]["clases"] == ["3A"]
    for pregunta in lista[0]["preguntas_json"]:
        assert "respuesta_correcta" not in pregunta
        assert "justificacion_pedagogica" not in pregunta


def test_docente_si_recibe_la_clave(cliente, clase_con_examen):
    lista = cliente.get("/api/cuestionarios", headers=clase_con_examen["docente"]).json()["data"]
    assert lista[0]["preguntas_json"][0]["respuesta_correcta"] == "a"


def test_entrega_calificada_en_el_servidor(cliente, clase_con_examen):
    d = clase_con_examen
    respuesta = entregar(cliente, d["estudiante"], d["examen_id"], {"0": "a", "1": "a"}, tiempo=42)
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["intento"]["puntos_obtenidos"] == 1
    assert datos["intento"]["puntos_totales"] == 4
    assert datos["intento"]["calificacion"] == 2.5
    assert datos["intento"]["tiempo_segundos"] == 42
    assert datos["respuestas_correctas"] == ["a", "b"]


def test_respuestas_a_preguntas_inexistentes_se_ignoran(cliente, clase_con_examen):
    d = clase_con_examen
    datos = entregar(cliente, d["estudiante"], d["examen_id"], {"0": "a", "7": "a", "x": "a"}).json()
    assert datos["intento"]["respuestas"] == {"0": "a"}


def test_tiempo_negativo_es_invalido(cliente, clase_con_examen):
    d = clase_con_examen
    assert entregar(cliente, d["estudiante"], d["examen_id"], {}, tiempo=-1).status_code == 422


def test_docente_no_puede_entregar(cliente, clase_con_examen):
    d = clase_con_examen
    assert entregar(cliente, d["docente"], d["examen_id"], {"0": "a"}).status_code == 403


def test_no_se_entrega_un_examen_despublicado(cliente, clase_con_examen):
    d = clase_con_examen
    cliente.put(f"/api/cuestionarios/{d['examen_id']}/despublicar", headers=d["docente"])
    assert entregar(cliente, d["estudiante"], d["examen_id"], {"0": "a"}).status_code == 404


def test_cada_reintento_es_un_intento_nuevo(cliente, clase_con_examen):
    d = clase_con_examen
    entregar(cliente, d["estudiante"], d["examen_id"], {"0": "b"})
    entregar(cliente, d["estudiante"], d["examen_id"], {"0": "a", "1": "b"})
    intentos = cliente.get(f"/api/cuestionarios/{d['examen_id']}/intentos", headers=d["docente"]).json()["data"]
    assert sorted(i["calificacion"] for i in intentos) == [0, 10]


def test_docente_ve_intentos_con_estudiante_y_clase(cliente, clase_con_examen):
    d = clase_con_examen
    entregar(cliente, d["estudiante"], d["examen_id"], {"0": "a"})
    datos = cliente.get(f"/api/cuestionarios/{d['examen_id']}/intentos", headers=d["docente"]).json()
    assert datos["clases"] == [{"id": d["clase"]["id"], "nombre": "3A"}]
    intento = datos["data"][0]
    assert intento["nombre_estudiante"] == "ana"
    assert intento["correo_estudiante"] == "ana@correo.com"
    assert intento["clase_ids"] == [d["clase"]["id"]]


def test_otro_docente_no_ve_los_intentos(cliente, registrar, clase_con_examen):
    otro, _ = registrar("otro", "docente")
    assert cliente.get(f"/api/cuestionarios/{clase_con_examen['examen_id']}/intentos", headers=otro).status_code == 403


def test_eliminar_examen_borra_sus_intentos(cliente, db, clase_con_examen):
    from database import ClaseCuestionario, Intento
    d = clase_con_examen
    entregar(cliente, d["estudiante"], d["examen_id"], {"0": "a"})
    assert cliente.delete(f"/api/cuestionarios/{d['examen_id']}", headers=d["docente"]).status_code == 200
    assert db.query(Intento).count() == 0
    assert db.query(ClaseCuestionario).count() == 0


# ---------- Retroalimentación y "Mis resultados" ----------

def test_entrega_devuelve_la_retroalimentacion(cliente, clase_con_examen):
    d = clase_con_examen
    datos = entregar(cliente, d["estudiante"], d["examen_id"], {"0": "a"}).json()
    assert datos["justificaciones"] == ["j1", "j2"]


def test_mis_intentos(cliente, clase_con_examen):
    d = clase_con_examen
    entregar(cliente, d["estudiante"], d["examen_id"], {"0": "b"})
    entregar(cliente, d["estudiante"], d["examen_id"], {"0": "a", "1": "b"})
    mios = cliente.get("/api/intentos/mios", headers=d["estudiante"]).json()["data"]
    assert [i["calificacion"] for i in mios] == [10, 0], "el más reciente primero"
    assert mios[0]["nombre_examen"] == "Examen" and mios[0]["materia"] == "Programación"


def test_mis_intentos_solo_los_propios(cliente, registrar, clase_con_examen):
    d = clase_con_examen
    entregar(cliente, d["estudiante"], d["examen_id"], {"0": "a"})
    beto, _ = registrar("beto")
    assert cliente.get("/api/intentos/mios", headers=beto).json()["data"] == []
    assert cliente.get("/api/intentos/mios", headers=d["docente"]).status_code == 403


def test_revisar_intento(cliente, registrar, clase_con_examen):
    d = clase_con_examen
    intento_id = entregar(cliente, d["estudiante"], d["examen_id"], {"0": "a"}).json()["intento"]["id"]
    revision = cliente.get(f"/api/intentos/{intento_id}", headers=d["estudiante"])
    assert revision.status_code == 200
    datos = revision.json()
    assert datos["intento"]["respuestas"] == {"0": "a"}
    assert datos["examen"]["preguntas_json"][1]["respuesta_correcta"] == "b"
    assert datos["examen"]["preguntas_json"][1]["justificacion_pedagogica"] == "j2"

    assert cliente.get(f"/api/intentos/{intento_id}", headers=d["docente"]).status_code == 200, "el docente dueño también"
    beto, _ = registrar("beto")
    otro, _ = registrar("otro", "docente")
    assert cliente.get(f"/api/intentos/{intento_id}", headers=beto).status_code == 403
    assert cliente.get(f"/api/intentos/{intento_id}", headers=otro).status_code == 403
    assert cliente.get("/api/intentos/99999", headers=d["estudiante"]).status_code == 404


# ---------- Seguridad de la respuesta (2.2) ----------

def test_la_seguridad_se_guarda_sin_cambiar_la_calificacion(cliente, clase_con_examen):
    d = clase_con_examen
    sin = entregar(cliente, d["estudiante"], d["examen_id"], {"0": "b", "1": "b"}).json()["intento"]
    con = cliente.post(f"/api/cuestionarios/{d['examen_id']}/intentos", headers=d["estudiante"], json={
        "respuestas": {"0": "b", "1": "b"},
        # "5" no existe y "x" no es un índice: se descartan
        "seguridad": {"0": True, "1": False, "5": True, "x": False},
    }).json()["intento"]
    assert con["calificacion"] == sin["calificacion"]
    assert sin["seguridad"] == {}
    assert con["seguridad"] == {"0": True, "1": False}


def test_seguridad_de_pregunta_sin_responder_se_descarta(cliente, clase_con_examen):
    d = clase_con_examen
    intento = cliente.post(f"/api/cuestionarios/{d['examen_id']}/intentos", headers=d["estudiante"],
                           json={"respuestas": {"0": "a"}, "seguridad": {"0": False, "1": True}}).json()["intento"]
    assert intento["seguridad"] == {"0": False}


def test_docente_y_revision_reciben_la_seguridad(cliente, clase_con_examen):
    d = clase_con_examen
    intento = cliente.post(f"/api/cuestionarios/{d['examen_id']}/intentos", headers=d["estudiante"],
                           json={"respuestas": {"0": "b"}, "seguridad": {"0": True}}).json()["intento"]
    docente = cliente.get(f"/api/cuestionarios/{d['examen_id']}/intentos", headers=d["docente"]).json()["data"]
    assert docente[0]["seguridad"] == {"0": True}
    assert cliente.get("/api/intentos/mios", headers=d["estudiante"]).json()["data"][0]["seguridad"] == {"0": True}
    revision = cliente.get(f"/api/intentos/{intento['id']}", headers=d["estudiante"]).json()
    assert revision["intento"]["seguridad"] == {"0": True}


def test_seguridad_invalida_se_rechaza(cliente, clase_con_examen):
    d = clase_con_examen
    respuesta = cliente.post(f"/api/cuestionarios/{d['examen_id']}/intentos", headers=d["estudiante"],
                             json={"respuestas": {"0": "a"}, "seguridad": {"0": "tal vez"}})
    assert respuesta.status_code == 422
