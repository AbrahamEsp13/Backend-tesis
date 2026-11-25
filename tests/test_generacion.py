"""Generación de cuestionarios (con la IA simulada), complejidad, preferencias y migración."""
import pytest
from sqlalchemy import create_engine, inspect, text

import generacion
from database import agregar_columnas_faltantes

PDF = ("material.pdf", b"%PDF-1.4 contenido de prueba", "application/pdf")


def generar(cliente, encabezados, **datos):
    formulario = {"num_preguntas": 3, "nombre_examen": "Parcial", "materia": "Programación", **datos}
    return cliente.post("/api/generar-cuestionario", headers=encabezados, files={"archivo": PDF}, data=formulario)


# ---------- Prompt ----------

def test_prompt_basico():
    prompt = generacion.construir_prompt("CTX", 5, "basico")
    assert "BÁSICO" in prompt and "AVANZADO" not in prompt


def test_prompt_equilibrado():
    prompt = generacion.construir_prompt("CTX", 5, "equilibrado")
    assert "EQUILIBRADO" in prompt


def test_prompt_avanzado():
    prompt = generacion.construir_prompt("CTX", 5, "avanzado")
    assert "AVANZADO" in prompt and "BÁSICO" not in prompt


def test_prompt_no_menciona_bloom():
    for complejidad in generacion.COMPLEJIDADES:
        prompt = generacion.construir_prompt("CTX", 5, complejidad)
        assert "Bloom" not in prompt and "nivel_bloom" not in prompt


def test_complejidad_desconocida_cae_en_equilibrado():
    assert generacion.construir_prompt("CTX", 5, "rara") == generacion.construir_prompt("CTX", 5, "equilibrado")


# ---------- Validación de la salida del modelo ----------

def test_validar_preguntas_corrige_la_letra_y_descarta_invalidas():
    datos = {"preguntas": [
        {"pregunta": "ok", "opciones": ["a) uno", "b) dos"], "respuesta_correcta": "b"},
        {"pregunta": "sin respuesta valida", "opciones": ["a) uno", "b) dos"], "respuesta_correcta": "z) tres"},
        {"pregunta": "", "opciones": ["a) uno", "b) dos"], "respuesta_correcta": "a) uno"},
    ]}
    validas = generacion.validar_preguntas(datos, 5)
    assert len(validas) == 1
    assert validas[0]["respuesta_correcta"] == "b) dos"
    assert validas[0]["puntuacion"] == 1


def test_validar_preguntas_sin_ninguna_valida_lanza_error():
    with pytest.raises(generacion.RespuestaInvalida):
        generacion.validar_preguntas([], 5)


# ---------- Endpoint de generación ----------

def test_generacion_usa_la_complejidad_y_la_guarda(cliente, registrar, ia_simulada):
    docente, _ = registrar("docente", "docente")
    respuesta = generar(cliente, docente, complejidad="basico")
    assert respuesta.status_code == 200
    assert respuesta.json()["creditos_disponibles"] == 4
    assert ia_simulada[-1] == {"num_preguntas": 3, "complejidad": "basico"}
    examen = cliente.get("/api/cuestionarios", headers=docente).json()["data"][0]
    assert examen["complejidad"] == "basico"


def test_generacion_sin_complejidad_usa_equilibrado(cliente, registrar, ia_simulada):
    docente, _ = registrar("docente", "docente")
    generar(cliente, docente)
    assert ia_simulada[-1]["complejidad"] == "equilibrado"


def test_generacion_rechaza_entradas_invalidas(cliente, registrar, ia_simulada):
    docente, _ = registrar("docente", "docente")
    estudiante, _ = registrar("ana")
    assert generar(cliente, docente, complejidad="dificil").status_code == 422
    assert generar(cliente, docente, num_preguntas=16).status_code == 422
    assert generar(cliente, estudiante).status_code == 403
    no_pdf = cliente.post("/api/generar-cuestionario", headers=docente,
                          files={"archivo": ("notas.txt", b"hola", "text/plain")},
                          data={"num_preguntas": 3, "nombre_examen": "x", "materia": "y"})
    assert no_pdf.status_code == 400
    assert ia_simulada == [], "nada de lo anterior debe llegar a la IA"


def test_creditos_diarios_se_agotan(cliente, registrar, ia_simulada):
    docente, _ = registrar("docente", "docente")
    for _ in range(5):
        assert generar(cliente, docente).status_code == 200
    agotado = generar(cliente, docente)
    assert agotado.status_code == 403
    assert "créditos" in agotado.json()["detail"]


def test_la_copia_conserva_la_complejidad(cliente, registrar, ia_simulada):
    docente, _ = registrar("docente", "docente")
    generar(cliente, docente, complejidad="avanzado")
    examen_id = cliente.get("/api/cuestionarios", headers=docente).json()["data"][0]["id"]
    cliente.put(f"/api/cuestionarios/{examen_id}/compartir", headers=docente)
    otro, _ = registrar("otro", "docente")
    assert cliente.post(f"/api/cuestionarios/{examen_id}/clonar", headers=otro).status_code == 200
    copia = cliente.get("/api/cuestionarios", headers=otro).json()["data"][0]
    assert copia["nombre_examen"] == "Parcial (Copia)"
    assert copia["complejidad"] == "avanzado"


# ---------- Preferencias ----------

def test_preferencia_de_complejidad(cliente, registrar):
    docente, usuario = registrar("docente", "docente")
    assert usuario["complejidad_preferida"] == "equilibrado"
    ruta = f"/api/usuarios/{usuario['id']}/preferencias"
    assert cliente.put(ruta, headers=docente, json={"complejidad_preferida": "avanzado"}).status_code == 200
    login = cliente.post("/api/auth/login", json={"correo": "docente@correo.com", "password": "secreta123"}).json()
    assert login["usuario"]["complejidad_preferida"] == "avanzado"
    assert cliente.put(ruta, headers=docente, json={"complejidad_preferida": "dificil"}).status_code == 422


def test_estudiante_no_tiene_preferencias_ia(cliente, registrar):
    estudiante, usuario = registrar("ana")
    ruta = f"/api/usuarios/{usuario['id']}/preferencias"
    assert cliente.put(ruta, headers=estudiante, json={"complejidad_preferida": "basico"}).status_code == 403


# ---------- Migración de columnas ----------

def test_migracion_agrega_columnas_a_una_base_antigua(tmp_path):
    motor = create_engine(f"sqlite:///{(tmp_path / 'antigua.db').as_posix()}")
    with motor.begin() as conexion:
        conexion.execute(text("CREATE TABLE usuarios (id INTEGER PRIMARY KEY, nombre VARCHAR)"))
        conexion.execute(text("CREATE TABLE cuestionarios_v3 (id INTEGER PRIMARY KEY, nombre_examen VARCHAR)"))
        conexion.execute(text("INSERT INTO cuestionarios_v3 (id, nombre_examen) VALUES (1, 'viejo')"))

    agregar_columnas_faltantes(motor)
    agregar_columnas_faltantes(motor)  # una segunda vez no debe fallar

    columnas = lambda tabla: {c["name"] for c in inspect(motor).get_columns(tabla)}
    assert "complejidad_preferida" in columnas("usuarios")
    assert "complejidad" in columnas("cuestionarios_v3")
    with motor.connect() as conexion:
        assert conexion.execute(text("SELECT nombre_examen FROM cuestionarios_v3")).scalar() == "viejo"
    motor.dispose()
