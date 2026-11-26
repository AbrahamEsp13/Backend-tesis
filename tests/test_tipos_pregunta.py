"""4.2 Más tipos de pregunta: verdadero/falso y respuesta corta calificada por la IA."""
import pytest

import calificacion
import generacion
import practicas
import respuestas_cortas
import versiones
from database import Cuestionario, Intento

VF = {"tipo": "verdadero_falso", "pregunta": "El sol es una estrella.", "opciones": ["Verdadero", "Falso"],
      "respuesta_correcta": "Verdadero", "justificacion_pedagogica": "Lo dice el texto.", "puntuacion": 1}
CORTA = {"tipo": "respuesta_corta", "pregunta": "¿Qué define una clase?", "opciones": [],
         "respuesta_correcta": "Los atributos y métodos de sus objetos", "justificacion_pedagogica": "Debe mencionar atributos y métodos.",
         "puntuacion": 2}
MULTIPLE = {"pregunta": "p", "opciones": ["a", "b"], "respuesta_correcta": "a", "justificacion_pedagogica": "j", "puntuacion": 1}


# ---------- Generación ----------

def test_prompt_por_tipos():
    solo = generacion.construir_prompt("ctx", 3)
    assert "opción múltiple" in solo and "verdadero_falso" not in solo
    mixto = generacion.construir_prompt("ctx", 4, "basico", ["respuesta_corta", "verdadero_falso", "raro"])
    assert '"verdadero_falso"' in mixto and '"respuesta_corta"' in mixto and '"opcion_multiple" (' not in mixto


def test_validar_tipos():
    datos = [
        {"tipo": "verdadero_falso", "pregunta": "a", "opciones": ["x"], "respuesta_correcta": "F"},
        {"tipo": "verdadero_falso", "pregunta": "b", "opciones": [], "respuesta_correcta": "quizá"},  # sin clave válida
        {"tipo": "respuesta_corta", "pregunta": "c", "opciones": ["a) x"], "respuesta_correcta": "  modelo  "},
        {"tipo": "respuesta_corta", "pregunta": "d", "opciones": [], "respuesta_correcta": ""},  # sin respuesta modelo
        {"tipo": "opcion_multiple", "pregunta": "e", "opciones": ["a) 1", "b) 2"], "respuesta_correcta": "b"},
    ]
    validas = generacion.validar_preguntas(datos, 10, ["verdadero_falso", "respuesta_corta", "opcion_multiple"])
    assert [(p["tipo"], p["opciones"], p["respuesta_correcta"]) for p in validas] == [
        ("verdadero_falso", ["Verdadero", "Falso"], "Falso"),
        ("respuesta_corta", [], "modelo"),
        ("opcion_multiple", ["a) 1", "b) 2"], "b) 2"),
    ]
    # Un tipo no pedido se descarta
    assert [p["pregunta"] for p in generacion.validar_preguntas(datos, 10, ["verdadero_falso"])] == ["a"]


def test_generar_con_tipos(cliente, registrar, ia_simulada):
    docente, _ = registrar("docente", "docente")
    cliente.post("/api/generar-cuestionario", headers=docente, files={"archivo": ("m.pdf", b"%PDF-1.4 x", "application/pdf")},
                 data={"num_preguntas": 2, "nombre_examen": "P", "materia": "M", "tipos": "respuesta_corta,verdadero_falso,otro"})
    assert ia_simulada[-1]["tipos"] == ["verdadero_falso", "respuesta_corta"]


def test_redondear_puntaje():
    assert [generacion.redondear_puntaje(v) for v in (0.9, 0.6, 0.2, "1", None, 7)] == [1.0, 0.5, 0.0, 1.0, 0.0, 1.0]


# ---------- Calificación ----------

def test_calificar_con_cortas():
    preguntas = [VF, CORTA, MULTIPLE]
    resultado = calificacion.calificar(preguntas, {"0": "Verdadero", "1": "texto", "2": "b"}, {"1": {"puntaje": 0.5}})
    assert resultado["puntos_obtenidos"] == 2.0 and resultado["puntos_totales"] == 4  # 1 (VF) + 0.5 * 2 (corta)
    assert resultado["respuestas_correctas"][1] == CORTA["respuesta_correcta"]


def test_corregir_con_la_ia_y_pendiente_si_falla(monkeypatch):
    llamadas = []

    def calificar_ia(items):
        llamadas.append(items)
        return {items[0]["numero"]: {"puntaje": 1.0, "comentario": "Bien."}}
    monkeypatch.setattr(generacion, "calificar_respuestas_cortas", calificar_ia)
    preguntas = [CORTA, CORTA, CORTA, MULTIPLE]
    cortas = respuestas_cortas.corregir(preguntas, {"0": " atributos y métodos ", "1": "", "2": "otra cosa", "3": "a"})
    assert len(llamadas) == 1 and [i["numero"] for i in llamadas[0]] == [1, 3]  # una sola llamada; la vacía no va
    assert cortas["0"] == {"texto": "atributos y métodos", "puntaje": 1.0, "comentario": "Bien.", "estado": "ia"}
    assert cortas["1"]["estado"] == "vacia" and cortas["1"]["puntaje"] == 0
    assert cortas["2"]["estado"] == "pendiente"  # la IA no la devolvió
    assert "3" not in cortas

    def falla(items):
        raise RuntimeError("sin servicio")
    monkeypatch.setattr(generacion, "calificar_respuestas_cortas", falla)
    assert respuestas_cortas.corregir([CORTA], {"0": "algo"})["0"]["estado"] == "pendiente"


# ---------- Flujo: entregar, revisar y corregir ----------

@pytest.fixture
def examen_mixto(clase_con_examen, db, monkeypatch):
    examen = db.query(Cuestionario).first()
    examen.preguntas_json = [VF, CORTA]
    db.commit()
    monkeypatch.setattr(generacion, "calificar_respuestas_cortas",
                        lambda items: {i["numero"]: {"puntaje": 0.5, "comentario": "Te faltó mencionar los métodos."} for i in items})
    return clase_con_examen


def entregar(cliente, datos, respuestas):
    respuesta = cliente.post(f"/api/cuestionarios/{datos['examen_id']}/intentos", headers=datos["estudiante"], json={"respuestas": respuestas})
    assert respuesta.status_code == 200, respuesta.text
    return respuesta.json()


def test_entregar_y_corregir(cliente, examen_mixto, db):
    resultado = entregar(cliente, examen_mixto, {"0": "Verdadero", "1": "Los atributos"})
    intento = resultado["intento"]
    assert intento["puntos_obtenidos"] == 2.0 and intento["puntos_totales"] == 3  # 1 + 0.5 * 2
    assert intento["respuestas"]["1"] == "Los atributos"
    assert intento["respuestas_cortas"]["1"]["comentario"] == "Te faltó mencionar los métodos."
    assert resultado["respuestas_correctas"][1] == CORTA["respuesta_correcta"]

    # El docente le da el punto completo
    ruta = f"/api/intentos/{intento['id']}/respuestas-cortas/1"
    corregido = cliente.put(ruta, headers=examen_mixto["docente"], json={"puntaje": 1, "comentario": "Correcto."})
    assert corregido.status_code == 200, corregido.text
    nuevo = corregido.json()["intento"]
    assert nuevo["puntos_obtenidos"] == 3 and nuevo["calificacion"] == 10
    assert nuevo["respuestas_cortas"]["1"] == {"texto": "Los atributos", "puntaje": 1, "comentario": "Correcto.", "estado": "docente"}

    # Solo el docente dueño y solo en preguntas cortas
    assert cliente.put(ruta, headers=examen_mixto["estudiante"], json={"puntaje": 1}).status_code == 403
    assert cliente.put(f"/api/intentos/{intento['id']}/respuestas-cortas/0", headers=examen_mixto["docente"], json={"puntaje": 1}).status_code == 400
    assert cliente.put(ruta, headers=examen_mixto["docente"], json={"puntaje": 0.7}).status_code == 422


def test_respuestas_ocultas_no_muestran_la_correccion(cliente, examen_mixto):
    cliente.put(f"/api/cuestionarios/{examen_mixto['examen_id']}/configuracion", headers=examen_mixto["docente"], json={"mostrar_respuestas": "nunca"})
    intento = entregar(cliente, examen_mixto, {"0": "Falso", "1": "algo"})["intento"]
    assert intento["respuestas_cortas"]["1"] == {"texto": "algo", "estado": "oculta"}


def test_practica_cuenta_las_cortas_incompletas():
    falladas = practicas.preguntas_falladas([VF, CORTA, CORTA], {"0": "Verdadero"}, {"1": {"puntaje": 1}, "2": {"puntaje": 0.5}})
    assert falladas == [2]


def test_versiones_no_barajan_verdadero_falso():
    orden = versiones.generar_orden([VF, CORTA, MULTIPLE], False, True)
    assert list(orden["opciones"]) == ["2"]


def test_hoja_en_papel_con_corta_calificada_por_el_docente(cliente, examen_mixto, db):
    alumna = examen_mixto["usuario_estudiante"]["id"]
    respuesta = cliente.post(f"/api/cuestionarios/{examen_mixto['examen_id']}/hojas/registrar", headers=examen_mixto["docente"],
                             json={"estudiante_id": alumna, "respuestas": {"0": 0}, "cortas": {"1": 0.5}})
    assert respuesta.status_code == 200, respuesta.text
    intento = respuesta.json()["intento"]
    assert intento["puntos_obtenidos"] == 2.0 and intento["respuestas_cortas"]["1"]["estado"] == "docente"


def test_entrega_automatica_califica_las_cortas(cliente, examen_mixto, db):
    import datetime
    from database import InicioIntento, ahora_utc
    cliente.put(f"/api/cuestionarios/{examen_mixto['examen_id']}/configuracion", headers=examen_mixto["docente"], json={"tiempo_limite": 5})
    inicio_id = cliente.post(f"/api/cuestionarios/{examen_mixto['examen_id']}/inicio", headers=examen_mixto["estudiante"]).json()["inicio_id"]
    cliente.put(f"/api/inicios/{inicio_id}/borrador", headers=examen_mixto["estudiante"], json={"respuestas": {"1": "atributos"}})
    inicio = db.query(InicioIntento).first()
    inicio.fecha_inicio = ahora_utc() - datetime.timedelta(minutes=10)
    db.commit()
    intentos = cliente.get(f"/api/cuestionarios/{examen_mixto['examen_id']}/intentos", headers=examen_mixto["docente"]).json()["data"]
    assert intentos[0]["entrega_automatica"] and intentos[0]["respuestas_cortas"]["1"]["puntaje"] == 0.5
    assert db.query(Intento).first().puntos_obtenidos == 1.0
