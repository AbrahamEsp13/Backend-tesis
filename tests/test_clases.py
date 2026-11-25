"""Clases: creación, código de acceso, asignación de exámenes y visibilidad para estudiantes."""
import re


def examenes_visibles(cliente, encabezados):
    return cliente.get("/api/cuestionarios", headers=encabezados).json()["data"]


def test_crear_clase_genera_codigo(cliente, registrar):
    docente, _ = registrar("docente", "docente")
    respuesta = cliente.post("/api/clases", headers=docente, json={"nombre": "  3A  ", "descripcion": ""})
    clase = respuesta.json()["clase"]
    assert clase["nombre"] == "3A"
    assert clase["descripcion"] is None
    assert re.fullmatch(r"[A-HJ-NP-Z2-9]{6}", clase["codigo"]), "sin 0/O/1/I/L"


def test_crear_clase_validaciones(cliente, registrar):
    docente, _ = registrar("docente", "docente")
    estudiante, _ = registrar("ana")
    assert cliente.post("/api/clases", headers=docente, json={"nombre": "   "}).status_code == 400
    assert cliente.post("/api/clases", headers=estudiante, json={"nombre": "x"}).status_code == 403


def test_estudiante_sin_clase_no_ve_ni_entrega(cliente, registrar, crear_cuestionario):
    _, docente = registrar("docente", "docente")
    estudiante, _ = registrar("ana")
    examen_id = crear_cuestionario(docente["id"])
    assert examenes_visibles(cliente, estudiante) == []
    entrega = cliente.post(f"/api/cuestionarios/{examen_id}/intentos", headers=estudiante, json={"respuestas": {}})
    assert entrega.status_code == 404


def test_solo_ven_el_examen_los_de_la_clase(cliente, registrar, clase_con_examen):
    beto, _ = registrar("beto")
    assert len(examenes_visibles(cliente, clase_con_examen["estudiante"])) == 1
    assert examenes_visibles(cliente, beto) == []


def test_borrador_asignado_no_es_visible(cliente, clase_con_examen):
    d = clase_con_examen
    cliente.put(f"/api/cuestionarios/{d['examen_id']}/despublicar", headers=d["docente"])
    assert examenes_visibles(cliente, d["estudiante"]) == []


def test_unirse_normaliza_el_codigo(cliente, registrar):
    docente, _ = registrar("docente", "docente")
    estudiante, _ = registrar("ana")
    codigo = cliente.post("/api/clases", headers=docente, json={"nombre": "3A"}).json()["clase"]["codigo"]
    escrito = f" {codigo[:3].lower()}-{codigo[3:]} "
    assert cliente.post("/api/clases/unirse", headers=estudiante, json={"codigo": escrito}).status_code == 200


def test_unirse_errores(cliente, registrar, clase_con_examen):
    d = clase_con_examen
    codigo = d["clase"]["codigo"]
    assert cliente.post("/api/clases/unirse", headers=d["estudiante"], json={"codigo": codigo}).status_code == 409
    assert cliente.post("/api/clases/unirse", headers=d["estudiante"], json={"codigo": "ZZZZZZ"}).status_code == 404
    assert cliente.post("/api/clases/unirse", headers=d["docente"], json={"codigo": codigo}).status_code == 403


def test_listado_de_clases_por_rol(cliente, clase_con_examen):
    d = clase_con_examen
    del_docente = cliente.get("/api/clases", headers=d["docente"]).json()["data"][0]
    assert del_docente["num_estudiantes"] == 1 and del_docente["num_cuestionarios"] == 1
    del_estudiante = cliente.get("/api/clases", headers=d["estudiante"]).json()["data"][0]
    assert del_estudiante["docente"] == "docente"
    assert "codigo" not in del_estudiante


def test_detalle_de_clase(cliente, registrar, clase_con_examen):
    d = clase_con_examen
    detalle = cliente.get(f"/api/clases/{d['clase']['id']}", headers=d["docente"]).json()
    assert [e["nombre"] for e in detalle["estudiantes"]] == ["ana"]
    assert [c["id"] for c in detalle["cuestionarios"]] == [d["examen_id"]]
    otro, _ = registrar("otro", "docente")
    assert cliente.get(f"/api/clases/{d['clase']['id']}", headers=otro).status_code == 403


def test_otro_docente_no_asigna_examenes_ajenos(cliente, registrar, clase_con_examen):
    d = clase_con_examen
    otro, _ = registrar("otro", "docente")
    clase_otro = cliente.post("/api/clases", headers=otro, json={"nombre": "B"}).json()["clase"]
    assert cliente.post(f"/api/clases/{d['clase']['id']}/cuestionarios/{d['examen_id']}", headers=otro).status_code == 403
    assert cliente.post(f"/api/clases/{clase_otro['id']}/cuestionarios/{d['examen_id']}", headers=otro).status_code == 403


def test_asignar_dos_veces_no_duplica(cliente, clase_con_examen):
    d = clase_con_examen
    ruta = f"/api/clases/{d['clase']['id']}/cuestionarios/{d['examen_id']}"
    assert cliente.post(ruta, headers=d["docente"]).status_code == 200
    detalle = cliente.get(f"/api/clases/{d['clase']['id']}", headers=d["docente"]).json()
    assert len(detalle["cuestionarios"]) == 1


def test_quitar_asignacion_oculta_el_examen(cliente, clase_con_examen):
    d = clase_con_examen
    cliente.delete(f"/api/clases/{d['clase']['id']}/cuestionarios/{d['examen_id']}", headers=d["docente"])
    assert examenes_visibles(cliente, d["estudiante"]) == []


def test_quitar_estudiante_y_salir_de_la_clase(cliente, registrar, clase_con_examen):
    d = clase_con_examen
    cliente.delete(f"/api/clases/{d['clase']['id']}/estudiantes/{d['usuario_estudiante']['id']}", headers=d["docente"])
    assert examenes_visibles(cliente, d["estudiante"]) == []

    beto, _ = registrar("beto")
    cliente.post("/api/clases/unirse", headers=beto, json={"codigo": d["clase"]["codigo"]})
    assert len(examenes_visibles(cliente, beto)) == 1
    cliente.delete(f"/api/clases/{d['clase']['id']}/salir", headers=beto)
    assert examenes_visibles(cliente, beto) == []


def test_regenerar_codigo_invalida_el_anterior(cliente, registrar, clase_con_examen):
    d = clase_con_examen
    nuevo = cliente.put(f"/api/clases/{d['clase']['id']}/codigo", headers=d["docente"]).json()["codigo"]
    assert nuevo != d["clase"]["codigo"]
    beto, _ = registrar("beto")
    assert cliente.post("/api/clases/unirse", headers=beto, json={"codigo": d["clase"]["codigo"]}).status_code == 404
    assert cliente.post("/api/clases/unirse", headers=beto, json={"codigo": nuevo}).status_code == 200
    assert len(examenes_visibles(cliente, d["estudiante"])) == 1, "los ya inscritos se quedan"


def test_editar_clase_conserva_el_codigo(cliente, registrar, clase_con_examen):
    d = clase_con_examen
    ruta = f"/api/clases/{d['clase']['id']}"
    editada = cliente.put(ruta, headers=d["docente"], json={"nombre": " 3A Tarde ", "descripcion": "Grupo B"}).json()["clase"]
    assert editada["nombre"] == "3A Tarde" and editada["descripcion"] == "Grupo B"
    assert editada["codigo"] == d["clase"]["codigo"]
    assert cliente.put(ruta, headers=d["docente"], json={"nombre": "  "}).status_code == 400
    otro, _ = registrar("otro", "docente")
    assert cliente.put(ruta, headers=otro, json={"nombre": "x"}).status_code == 403


def test_eliminar_clase_conserva_examenes_e_intentos(cliente, db, clase_con_examen):
    from database import Cuestionario, Intento
    d = clase_con_examen
    cliente.post(f"/api/cuestionarios/{d['examen_id']}/intentos", headers=d["estudiante"], json={"respuestas": {"0": "a"}})
    assert cliente.delete(f"/api/clases/{d['clase']['id']}", headers=d["docente"]).status_code == 200
    assert cliente.get("/api/clases", headers=d["docente"]).json()["data"] == []
    assert db.query(Cuestionario).count() == 1
    assert db.query(Intento).count() == 1
    assert examenes_visibles(cliente, d["estudiante"]) == []


# ---------- Publicar y asignar en un paso ----------

def test_publicar_asigna_a_las_clases_elegidas(cliente, registrar, crear_cuestionario, clase_con_examen):
    d = clase_con_examen
    borrador = crear_cuestionario(d["usuario_docente"]["id"], nombre="Nuevo", publicado=False)
    otra = cliente.post("/api/clases", headers=d["docente"], json={"nombre": "3B"}).json()["clase"]
    respuesta = cliente.put(f"/api/cuestionarios/{borrador}/publicar", headers=d["docente"],
                            json={"clases": [d["clase"]["id"], otra["id"]]})
    assert respuesta.status_code == 200
    assert respuesta.json()["clases"] == sorted([d["clase"]["id"], otra["id"]])
    assert {e["id"] for e in examenes_visibles(cliente, d["estudiante"])} == {d["examen_id"], borrador}
    asignadas = cliente.get(f"/api/cuestionarios/{borrador}/clases", headers=d["docente"]).json()["clases"]
    assert asignadas == sorted([d["clase"]["id"], otra["id"]])


def test_publicar_quita_las_clases_desmarcadas(cliente, registrar, clase_con_examen):
    d = clase_con_examen
    otra = cliente.post("/api/clases", headers=d["docente"], json={"nombre": "3B"}).json()["clase"]
    cliente.put(f"/api/cuestionarios/{d['examen_id']}/publicar", headers=d["docente"], json={"clases": [otra["id"]]})
    assert examenes_visibles(cliente, d["estudiante"]) == []
    assert cliente.get(f"/api/cuestionarios/{d['examen_id']}/clases", headers=d["docente"]).json()["clases"] == [otra["id"]]


def test_publicar_sin_cuerpo_conserva_las_asignaciones(cliente, clase_con_examen):
    d = clase_con_examen
    cliente.put(f"/api/cuestionarios/{d['examen_id']}/despublicar", headers=d["docente"])
    assert cliente.put(f"/api/cuestionarios/{d['examen_id']}/publicar", headers=d["docente"]).status_code == 200
    assert [e["id"] for e in examenes_visibles(cliente, d["estudiante"])] == [d["examen_id"]]


def test_publicar_en_clase_ajena_no_cambia_nada(cliente, registrar, crear_cuestionario, clase_con_examen):
    d = clase_con_examen
    otro, _ = registrar("otro", "docente")
    clase_otro = cliente.post("/api/clases", headers=otro, json={"nombre": "B"}).json()["clase"]
    borrador = crear_cuestionario(d["usuario_docente"]["id"], nombre="Nuevo", publicado=False)
    respuesta = cliente.put(f"/api/cuestionarios/{borrador}/publicar", headers=d["docente"],
                            json={"clases": [d["clase"]["id"], clase_otro["id"]]})
    assert respuesta.status_code == 403
    assert cliente.get(f"/api/cuestionarios/{borrador}/clases", headers=d["docente"]).json()["clases"] == []
    assert [e["id"] for e in examenes_visibles(cliente, d["estudiante"])] == [d["examen_id"]]
    assert cliente.put(f"/api/cuestionarios/{borrador}/publicar", headers=d["docente"], json={"clases": [999]}).status_code == 404


def test_clases_del_cuestionario_solo_para_su_docente(cliente, registrar, clase_con_examen):
    d = clase_con_examen
    otro, _ = registrar("otro", "docente")
    assert cliente.get(f"/api/cuestionarios/{d['examen_id']}/clases", headers=otro).status_code == 403
    assert cliente.get(f"/api/cuestionarios/{d['examen_id']}/clases", headers=d["estudiante"]).status_code == 403
