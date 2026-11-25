"""2.1 Retroalimentación con cita al documento: páginas en el RAG, validación de la cita y privacidad."""
import generacion
import rag
from database import Cuestionario

FRAGMENTOS = [
    {"numero": 1, "texto": "La herencia permite que una clase reutilice el código de otra.", "pagina": 2, "pagina_fin": 2},
    {"numero": 2, "texto": "El encapsulamiento oculta el estado interno\nde un objeto.", "pagina": 3, "pagina_fin": 4},
]
PDF = ("material.pdf", b"%PDF-1.4 contenido de prueba", "application/pdf")


# ---------- RAG: página de cada fragmento ----------

def test_pagina_de_cada_posicion():
    texto, inicios = rag.unir_paginas(["uno", "dos dos", "tres"])
    assert texto == "uno\ndos dos\ntres"
    assert [rag.pagina_de(texto.index(p), inicios) for p in ("uno", "dos", "tres")] == [1, 2, 3]


def test_fragmentos_conservan_su_posicion():
    paginas = [" ".join(["alfa"] * 300), " ".join(["beta"] * 300)]  # ~1500 caracteres cada una
    texto, inicios = rag.unir_paginas(paginas)
    fragmentos = rag.fragmentar_con_posiciones(texto)
    assert [f for _, _, f in fragmentos] == rag.fragmentar(texto)
    for inicio, fin, fragmento in fragmentos:
        assert texto[inicio:fin] == fragmento
    primera, ultima = fragmentos[0], fragmentos[-1]
    assert rag.pagina_de(primera[0], inicios) == 1
    assert rag.pagina_de(ultima[1] - 1, inicios) == 2
    # Algún fragmento cruza de la página 1 a la 2
    assert any(rag.pagina_de(i, inicios) == 1 and rag.pagina_de(f - 1, inicios) == 2 for i, f, _ in fragmentos)


# ---------- Validación de la cita ----------

def test_cita_textual_en_el_fragmento_indicado():
    fuente = generacion.buscar_fuente(2, "oculta el estado interno de un objeto", FRAGMENTOS)
    assert fuente == {"texto": "oculta el estado interno de un objeto", "pagina": 3, "pagina_fin": 4, "literal": True}


def test_cita_en_otro_fragmento_usa_las_paginas_correctas():
    fuente = generacion.buscar_fuente(2, "«permite que una clase reutilice el código»", FRAGMENTOS)
    assert fuente == {"texto": "permite que una clase reutilice el código", "pagina": 2, "pagina_fin": 2, "literal": True}


def test_cita_inventada_usa_el_fragmento_completo():
    fuente = generacion.buscar_fuente("1", "esto no aparece en el documento", FRAGMENTOS)
    assert fuente == {"texto": FRAGMENTOS[0]["texto"], "pagina": 2, "pagina_fin": 2, "literal": False}


def test_sin_cita_ni_fragmento_valido_no_hay_fuente():
    assert generacion.buscar_fuente(9, "inventada por completo", FRAGMENTOS) is None
    assert generacion.buscar_fuente(None, "", FRAGMENTOS) is None
    assert generacion.buscar_fuente("abc", "corta", FRAGMENTOS) is None


def test_agregar_fuentes_quita_los_campos_crudos():
    preguntas = generacion.validar_preguntas([
        {"pregunta": "p1", "opciones": ["a) x", "b) y"], "respuesta_correcta": "a) x",
         "fragmento": 1, "cita": "La herencia permite que una clase"},
        {"pregunta": "p2", "opciones": ["a) x", "b) y"], "respuesta_correcta": "b"},
    ], 5)
    con_fuente = generacion.agregar_fuentes(preguntas, FRAGMENTOS)
    assert con_fuente[0]["fuente"]["pagina"] == 2
    assert "fuente" not in con_fuente[1]
    for pregunta in con_fuente:
        assert "fragmento" not in pregunta and "cita" not in pregunta


# ---------- Endpoint y privacidad ----------

def test_la_generacion_guarda_la_fuente(cliente, registrar, ia_simulada):
    docente, _ = registrar("docente", "docente")
    respuesta = cliente.post("/api/generar-cuestionario", headers=docente, files={"archivo": PDF},
                             data={"num_preguntas": 1, "nombre_examen": "Parcial", "materia": "POO"})
    assert respuesta.status_code == 200
    pregunta = cliente.get("/api/cuestionarios", headers=docente).json()["data"][0]["preguntas_json"][0]
    assert pregunta["fuente"] == {"texto": "define los atributos y métodos", "pagina": 4, "pagina_fin": 5, "literal": True}
    assert "fragmento" not in pregunta and "cita" not in pregunta


def test_el_estudiante_ve_la_fuente_solo_al_entregar(cliente, db, clase_con_examen):
    d = clase_con_examen
    cuestionario = db.get(Cuestionario, d["examen_id"])
    preguntas = [dict(p) for p in cuestionario.preguntas_json]
    preguntas[0]["fuente"] = {"texto": "cita", "pagina": 7, "pagina_fin": 7}
    cuestionario.preguntas_json = preguntas
    db.commit()

    examen = cliente.get("/api/cuestionarios", headers=d["estudiante"]).json()["data"][0]
    assert all("fuente" not in p for p in examen["preguntas_json"])

    entrega = cliente.post(f"/api/cuestionarios/{d['examen_id']}/intentos", headers=d["estudiante"],
                           json={"respuestas": {"0": "b"}, "tiempo_segundos": 30}).json()
    assert entrega["fuentes"] == [{"texto": "cita", "pagina": 7, "pagina_fin": 7}, None]

    revision = cliente.get(f"/api/intentos/{entrega['intento']['id']}", headers=d["estudiante"]).json()
    assert revision["examen"]["preguntas_json"][0]["fuente"]["pagina"] == 7
