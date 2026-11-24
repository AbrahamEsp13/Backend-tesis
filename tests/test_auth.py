"""Registro, login, CORS, cambio de contraseña y eliminación de cuenta."""
import pytest

from conftest import PASSWORD
from database import Clase, ClaseCuestionario, ClaseEstudiante, Cuestionario, Intento


def registrar_datos(cliente, **cambios):
    datos = {"nombre": "Ana", "correo": "Ana@Correo.com", "password": PASSWORD, "rol": "estudiante", **cambios}
    return cliente.post("/api/auth/register", json=datos)


# ---------- Registro ----------

@pytest.mark.parametrize("cambios, fragmento", [
    ({"password": "corta"}, "8 caracteres"),
    ({"correo": "sin-arroba"}, "correo"),
    ({"nombre": "   "}, "nombre"),
])
def test_registro_rechaza_datos_invalidos_en_espanol(cliente, cambios, fragmento):
    respuesta = registrar_datos(cliente, **cambios)
    assert respuesta.status_code == 400
    assert fragmento in respuesta.json()["detail"]


def test_registro_rechaza_rol_desconocido(cliente):
    assert registrar_datos(cliente, rol="admin").status_code == 422


def test_correo_duplicado_sin_importar_mayusculas(cliente):
    assert registrar_datos(cliente).status_code == 200
    assert registrar_datos(cliente, correo="  ana@CORREO.com ").status_code == 400


# ---------- Login ----------

def test_login_sin_distinguir_mayusculas(cliente):
    registrar_datos(cliente)
    respuesta = cliente.post("/api/auth/login", json={"correo": "ANA@correo.com ", "password": PASSWORD})
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["token"]
    assert "password_hash" not in datos["usuario"]


def test_login_no_revela_si_el_correo_existe(cliente):
    registrar_datos(cliente)
    no_existe = cliente.post("/api/auth/login", json={"correo": "nadie@correo.com", "password": PASSWORD})
    mala = cliente.post("/api/auth/login", json={"correo": "ana@correo.com", "password": "otra-clave"})
    assert no_existe.status_code == mala.status_code == 401
    assert no_existe.json() == mala.json() == {"detail": "Correo o contraseña incorrectos."}


def test_sin_token_o_token_invalido_es_401(cliente):
    assert cliente.get("/api/cuestionarios").status_code == 401
    assert cliente.get("/api/cuestionarios", headers={"Authorization": "Bearer basura"}).status_code == 401


# ---------- CORS ----------

def preflight(cliente, origen):
    return cliente.options("/api/auth/login", headers={
        "Origin": origen, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type",
    })


def test_cors_permite_los_origenes_configurados(cliente):
    for origen in ("http://localhost:5173", "https://quizai-pruebas.vercel.app"):
        assert preflight(cliente, origen).headers.get("access-control-allow-origin") == origen


def test_cors_rechaza_otros_origenes(cliente):
    assert "access-control-allow-origin" not in preflight(cliente, "https://sitio-ajeno.com").headers


# ---------- Cambio de contraseña ----------

def cambiar(cliente, encabezados, usuario_id, actual, nueva):
    return cliente.put(f"/api/usuarios/{usuario_id}/password", headers=encabezados,
                       json={"password_actual": actual, "password_nueva": nueva})


def test_cambio_de_password(cliente, registrar):
    ana, usuario = registrar("ana")
    assert cambiar(cliente, ana, usuario["id"], PASSWORD, "nueva-clave-123").status_code == 200
    assert cliente.post("/api/auth/login", json={"correo": "ana@correo.com", "password": "nueva-clave-123"}).status_code == 200
    assert cliente.post("/api/auth/login", json={"correo": "ana@correo.com", "password": PASSWORD}).status_code == 401


def test_cambio_de_password_validaciones(cliente, registrar):
    ana, usuario = registrar("ana")
    beto, _ = registrar("beto")
    assert cambiar(cliente, ana, usuario["id"], "incorrecta", "nueva-clave-123").status_code == 400
    assert cambiar(cliente, ana, usuario["id"], PASSWORD, "corta").status_code == 422
    assert cambiar(cliente, ana, usuario["id"], PASSWORD, PASSWORD).status_code == 400
    assert cambiar(cliente, beto, usuario["id"], PASSWORD, "nueva-clave-123").status_code == 403


# ---------- Eliminación de cuenta ----------

def eliminar(cliente, encabezados, usuario_id, password):
    return cliente.request("DELETE", f"/api/usuarios/{usuario_id}", headers=encabezados, json={"password": password})


def test_eliminar_cuenta_exige_password(cliente, registrar):
    ana, usuario = registrar("ana")
    assert eliminar(cliente, ana, usuario["id"], "incorrecta").status_code == 400
    assert cliente.get("/api/clases", headers=ana).status_code == 200


def test_eliminar_estudiante_borra_sus_intentos_y_membresias(cliente, db, clase_con_examen):
    d = clase_con_examen
    cliente.post(f"/api/cuestionarios/{d['examen_id']}/intentos", headers=d["estudiante"], json={"respuestas": {"0": "a"}})
    estudiante_id = d["usuario_estudiante"]["id"]

    assert eliminar(cliente, d["estudiante"], estudiante_id, PASSWORD).status_code == 200
    assert db.query(Intento).filter(Intento.estudiante_id == estudiante_id).count() == 0
    assert db.query(ClaseEstudiante).filter(ClaseEstudiante.estudiante_id == estudiante_id).count() == 0
    assert cliente.get("/api/clases", headers=d["estudiante"]).status_code == 401, "el token de una cuenta borrada ya no sirve"


def test_eliminar_docente_borra_examenes_clases_e_intentos(cliente, db, clase_con_examen):
    d = clase_con_examen
    cliente.post(f"/api/cuestionarios/{d['examen_id']}/intentos", headers=d["estudiante"], json={"respuestas": {"0": "a"}})

    assert eliminar(cliente, d["docente"], d["usuario_docente"]["id"], PASSWORD).status_code == 200
    assert db.query(Cuestionario).count() == 0
    assert db.query(Clase).count() == db.query(ClaseCuestionario).count() == db.query(ClaseEstudiante).count() == 0
    assert db.query(Intento).count() == 0
    assert cliente.get("/api/cuestionarios", headers=d["estudiante"]).json()["data"] == []
