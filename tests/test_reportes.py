"""2.5 Reportes de preguntas: el estudiante reporta, el docente los ve y los descarta."""
from database import Cuestionario, Reporte
from conftest import PASSWORD


def entregar(cliente, d):
    return cliente.post(f"/api/cuestionarios/{d['examen_id']}/intentos", headers=d["estudiante"],
                        json={"respuestas": {"0": "b", "1": "b"}}).json()["intento"]


def reportar(cliente, encabezados, intento_id, **datos):
    cuerpo = {"pregunta": 0, "motivo": "clave_incorrecta", **datos}
    return cliente.post(f"/api/intentos/{intento_id}/reportes", headers=encabezados, json=cuerpo)


def reportes_del_docente(cliente, d):
    return cliente.get(f"/api/cuestionarios/{d['examen_id']}/intentos", headers=d["docente"]).json()["reportes"]


def test_el_docente_ve_el_reporte(cliente, clase_con_examen):
    d = clase_con_examen
    intento = entregar(cliente, d)
    assert reportar(cliente, d["estudiante"], intento["id"], comentario="  La b también es válida  ").status_code == 200
    [reporte] = reportes_del_docente(cliente, d)
    assert reporte["pregunta"] == 0
    assert reporte["texto_pregunta"] == "p1"
    assert reporte["motivo"] == "clave_incorrecta"
    assert reporte["comentario"] == "La b también es válida"
    assert reporte["nombre_estudiante"] == "ana"


def test_reportar_de_nuevo_actualiza_sin_duplicar(cliente, clase_con_examen):
    d = clase_con_examen
    intento = entregar(cliente, d)
    reportar(cliente, d["estudiante"], intento["id"])
    otro_intento = entregar(cliente, d)
    reportar(cliente, d["estudiante"], otro_intento["id"], motivo="ambigua", comentario="")
    [reporte] = reportes_del_docente(cliente, d)
    assert reporte["motivo"] == "ambigua" and reporte["comentario"] is None


def test_la_revision_indica_las_preguntas_ya_reportadas(cliente, clase_con_examen):
    d = clase_con_examen
    intento = entregar(cliente, d)
    reportar(cliente, d["estudiante"], intento["id"], pregunta=1)
    for encabezados in (d["estudiante"], d["docente"]):
        revision = cliente.get(f"/api/intentos/{intento['id']}", headers=encabezados).json()
        assert revision["preguntas_reportadas"] == [1]


def test_solo_el_dueno_del_intento_reporta(cliente, registrar, clase_con_examen):
    d = clase_con_examen
    intento = entregar(cliente, d)
    otro, _ = registrar("beto")
    assert reportar(cliente, otro, intento["id"]).status_code == 404
    assert reportar(cliente, d["docente"], intento["id"]).status_code == 404
    assert reportar(cliente, d["estudiante"], 999).status_code == 404


def test_reporte_invalido(cliente, clase_con_examen):
    d = clase_con_examen
    intento = entregar(cliente, d)
    assert reportar(cliente, d["estudiante"], intento["id"], pregunta=5).status_code == 400
    assert reportar(cliente, d["estudiante"], intento["id"], motivo="me cae mal").status_code == 422
    assert reportar(cliente, d["estudiante"], intento["id"], comentario="x" * 501).status_code == 422
    assert reportes_del_docente(cliente, d) == []


def test_el_docente_descarta_un_reporte(cliente, registrar, clase_con_examen):
    d = clase_con_examen
    intento = entregar(cliente, d)
    reportar(cliente, d["estudiante"], intento["id"])
    [reporte] = reportes_del_docente(cliente, d)
    ruta = f"/api/cuestionarios/{d['examen_id']}/reportes/{reporte['id']}"
    otro, _ = registrar("otro", "docente")
    assert cliente.delete(ruta, headers=otro).status_code == 403
    assert cliente.delete(ruta, headers=d["docente"]).status_code == 200
    assert reportes_del_docente(cliente, d) == []


def test_borrar_la_cuenta_del_estudiante_borra_sus_reportes(cliente, db, clase_con_examen):
    d = clase_con_examen
    intento = entregar(cliente, d)
    reportar(cliente, d["estudiante"], intento["id"])
    cliente.request("DELETE", f"/api/usuarios/{d['usuario_estudiante']['id']}", headers=d["estudiante"], json={"password": PASSWORD})
    assert db.query(Reporte).count() == 0


def test_borrar_examen_borra_sus_reportes(cliente, db, clase_con_examen):
    d = clase_con_examen
    intento = entregar(cliente, d)
    reportar(cliente, d["estudiante"], intento["id"])
    cliente.delete(f"/api/cuestionarios/{d['examen_id']}", headers=d["docente"])
    assert db.query(Reporte).count() == 0
    assert db.query(Cuestionario).count() == 0
