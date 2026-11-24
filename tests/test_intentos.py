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
