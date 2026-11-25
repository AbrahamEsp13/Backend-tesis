"""2.7 Calidad del generador: valoración de cada pregunta y estadísticas por complejidad."""
import valoraciones
from database import Cuestionario

PREGUNTAS = [
    {"pregunta": "p1", "opciones": ["a", "b"], "respuesta_correcta": "a", "justificacion_pedagogica": "j",
     "puntuacion": 1, "fuente": {"texto": "t", "pagina": 1, "pagina_fin": 1, "literal": True}},
    {"pregunta": "p2", "opciones": ["a", "b"], "respuesta_correcta": "b", "justificacion_pedagogica": "j",
     "puntuacion": 1, "fuente": {"texto": "t", "pagina": 2, "pagina_fin": 2, "literal": False}},
    {"pregunta": "p3", "opciones": ["a", "b"], "respuesta_correcta": "a", "justificacion_pedagogica": "j", "puntuacion": 1},
    {"pregunta": "manual", "opciones": ["a", "b"], "respuesta_correcta": "a", "justificacion_pedagogica": "j",
     "puntuacion": 1, "origen": "manual"},
]


def valorar(cliente, encabezados, examen_id, indice, **datos):
    return cliente.put(f"/api/cuestionarios/{examen_id}/preguntas/{indice}/valoracion", headers=encabezados, json=datos)


def preparar(cliente, registrar, crear_cuestionario, db, complejidad="avanzado"):
    docente, usuario = registrar("docente", "docente")
    examen_id = crear_cuestionario(usuario["id"], preguntas=PREGUNTAS)
    db.get(Cuestionario, examen_id).complejidad = complejidad
    db.commit()
    return docente, examen_id


def test_valorar_y_ver_estadisticas(cliente, registrar, crear_cuestionario, db):
    docente, examen_id = preparar(cliente, registrar, crear_cuestionario, db)
    assert valorar(cliente, docente, examen_id, 0, util=True).status_code == 200
    respuesta = valorar(cliente, docente, examen_id, 1, util=False, motivo="ambigua")
    assert respuesta.json()["valoracion"]["motivo"] == "ambigua"
    valorar(cliente, docente, examen_id, 3, util=False, motivo="otro")  # manual: no cuenta

    stats = cliente.get("/api/estadisticas/generador", headers=docente).json()
    total = stats["total"]
    assert total["generadas"] == 3
    assert total["con_fuente"] == 2 and total["con_cita_literal"] == 1
    assert (total["valoradas"], total["utiles"], total["no_utiles"]) == (2, 1, 1)
    assert total["motivos"]["ambigua"] == 1 and total["motivos"]["otro"] == 0
    assert stats["por_complejidad"]["avanzado"]["generadas"] == 3

    examen = cliente.get("/api/cuestionarios", headers=docente).json()["data"][0]
    assert examen["preguntas_json"][1]["valoracion"]["util"] is False


def test_util_ignora_el_motivo_y_null_quita_la_valoracion(cliente, registrar, crear_cuestionario, db):
    docente, examen_id = preparar(cliente, registrar, crear_cuestionario, db)
    assert valorar(cliente, docente, examen_id, 0, util=True, motivo="ambigua").json()["valoracion"]["motivo"] is None
    assert valorar(cliente, docente, examen_id, 0, util=None).json()["valoracion"] is None
    examen = cliente.get("/api/cuestionarios", headers=docente).json()["data"][0]
    assert "valoracion" not in examen["preguntas_json"][0]


def test_se_puede_valorar_un_examen_publicado(cliente, clase_con_examen):
    d = clase_con_examen  # el examen del fixture está publicado
    assert valorar(cliente, d["docente"], d["examen_id"], 0, util=True).status_code == 200


def test_permisos_y_validaciones(cliente, registrar, clase_con_examen):
    d = clase_con_examen
    otro, _ = registrar("otro", "docente")
    assert valorar(cliente, otro, d["examen_id"], 0, util=True).status_code == 403
    assert valorar(cliente, d["estudiante"], d["examen_id"], 0, util=True).status_code == 403
    assert valorar(cliente, d["docente"], d["examen_id"], 9, util=True).status_code == 404
    assert valorar(cliente, d["docente"], d["examen_id"], 0, util=False, motivo="fea").status_code == 422
    assert cliente.get("/api/estadisticas/generador", headers=d["estudiante"]).status_code == 403


def test_el_estudiante_no_recibe_la_valoracion(cliente, clase_con_examen):
    d = clase_con_examen
    valorar(cliente, d["docente"], d["examen_id"], 0, util=False, motivo="clave_incorrecta")
    examen = cliente.get("/api/cuestionarios", headers=d["estudiante"]).json()["data"][0]
    assert all("valoracion" not in p for p in examen["preguntas_json"])
    intento = cliente.post(f"/api/cuestionarios/{d['examen_id']}/intentos", headers=d["estudiante"],
                           json={"respuestas": {"0": "a"}}).json()["intento"]
    revision = cliente.get(f"/api/intentos/{intento['id']}", headers=d["estudiante"]).json()
    assert all("valoracion" not in p for p in revision["examen"]["preguntas_json"])
    revision_docente = cliente.get(f"/api/intentos/{intento['id']}", headers=d["docente"]).json()
    assert revision_docente["examen"]["preguntas_json"][0]["valoracion"]["util"] is False


def test_la_copia_no_hereda_las_valoraciones(cliente, registrar, clase_con_examen):
    d = clase_con_examen
    valorar(cliente, d["docente"], d["examen_id"], 0, util=True)
    cliente.put(f"/api/cuestionarios/{d['examen_id']}/compartir", headers=d["docente"])
    otro, _ = registrar("otro", "docente")
    cliente.post(f"/api/cuestionarios/{d['examen_id']}/clonar", headers=otro)
    copia = cliente.get("/api/cuestionarios", headers=otro).json()["data"][0]
    assert all("valoracion" not in p for p in copia["preguntas_json"])


def test_estadisticas_agrupan_examenes_sin_complejidad():
    class Examen:
        def __init__(self, complejidad, preguntas):
            self.complejidad, self.preguntas_json = complejidad, preguntas
    stats = valoraciones.estadisticas([Examen(None, PREGUNTAS[:1]), Examen("basico", PREGUNTAS[2:3])])
    assert stats["por_complejidad"][valoraciones.SIN_COMPLEJIDAD]["generadas"] == 1
    assert stats["por_complejidad"]["basico"]["generadas"] == 1
    assert stats["total"]["generadas"] == 2
