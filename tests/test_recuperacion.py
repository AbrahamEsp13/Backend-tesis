"""4.7 Recuperar la contraseña por correo."""
import datetime
import re

import pytest

import correo
import recuperacion
from conftest import PASSWORD
from database import RestablecimientoPassword, ahora_utc


@pytest.fixture
def buzon(monkeypatch):
    """Reemplaza el envío: guarda los correos en lugar de mandarlos."""
    enviados = []
    monkeypatch.setattr(correo, "enviar", lambda destinatario, asunto, texto, html=None: enviados.append(
        {"para": destinatario, "asunto": asunto, "texto": texto, "html": html}) or True)
    return enviados


def token_de(mensaje) -> str:
    return re.search(r"/restablecer\?token=([\w-]+)", mensaje["texto"]).group(1)


def pedir(cliente, correo_usuario="ana@correo.com"):
    return cliente.post("/api/auth/recuperar", json={"correo": correo_usuario})


def test_flujo_completo(cliente, registrar, buzon):
    registrar("ana", "estudiante")
    respuesta = pedir(cliente, "  ANA@correo.com ")
    assert respuesta.status_code == 200 and "Si hay una cuenta" in respuesta.json()["mensaje"]
    assert len(buzon) == 1 and buzon[0]["para"] == "ana@correo.com"
    assert "http://localhost:5173/restablecer?token=" in buzon[0]["texto"] and buzon[0]["html"]
    token = token_de(buzon[0])

    verificado = cliente.post("/api/auth/restablecer/verificar", json={"token": token}).json()
    assert verificado["correo"] == "an*@correo.com"

    cambio = cliente.post("/api/auth/restablecer", json={"token": token, "password_nueva": "nueva-clave-123"})
    assert cambio.status_code == 200, cambio.text
    assert cliente.post("/api/auth/login", json={"correo": "ana@correo.com", "password": PASSWORD}).status_code != 200
    assert cliente.post("/api/auth/login", json={"correo": "ana@correo.com", "password": "nueva-clave-123"}).status_code == 200
    # El enlace es de un solo uso
    assert cliente.post("/api/auth/restablecer", json={"token": token, "password_nueva": "otra-clave-456"}).status_code == 400


def test_misma_respuesta_si_el_correo_no_existe(cliente, buzon):
    respuesta = pedir(cliente, "nadie@correo.com")
    assert respuesta.status_code == 200 and respuesta.json()["mensaje"] == recuperacion.MENSAJE_ENVIADO
    assert buzon == []


def test_solo_se_guarda_el_hash(cliente, registrar, buzon, db):
    registrar("ana", "estudiante")
    pedir(cliente)
    registro = db.query(RestablecimientoPassword).first()
    assert registro.token_hash == recuperacion.hash_token(token_de(buzon[0])) and token_de(buzon[0]) not in registro.token_hash


def test_enlace_vencido_o_reemplazado(cliente, registrar, buzon, db):
    registrar("ana", "estudiante")
    pedir(cliente)
    pedir(cliente)
    viejo, nuevo = token_de(buzon[0]), token_de(buzon[1])
    assert cliente.post("/api/auth/restablecer/verificar", json={"token": viejo}).status_code == 400  # lo reemplazó el nuevo

    registro = db.query(RestablecimientoPassword).filter(RestablecimientoPassword.token_hash == recuperacion.hash_token(nuevo)).first()
    registro.expira = ahora_utc() - datetime.timedelta(minutes=1)
    db.commit()
    respuesta = cliente.post("/api/auth/restablecer", json={"token": nuevo, "password_nueva": "nueva-clave-123"})
    assert respuesta.status_code == 400 and "venció" in respuesta.json()["detail"]


def test_limite_de_solicitudes(cliente, registrar, buzon):
    registrar("ana", "estudiante")
    for _ in range(recuperacion.SOLICITUDES_POR_HORA + 2):
        assert pedir(cliente).status_code == 200
    assert len(buzon) == recuperacion.SOLICITUDES_POR_HORA


def test_cierra_las_sesiones_abiertas(cliente, registrar, buzon):
    sesion_vieja, usuario = registrar("ana", "estudiante")
    assert cliente.get("/api/intentos/mios", headers=sesion_vieja).status_code == 200
    pedir(cliente)
    cliente.post("/api/auth/restablecer", json={"token": token_de(buzon[0]), "password_nueva": "nueva-clave-123"})
    assert cliente.get("/api/intentos/mios", headers=sesion_vieja).status_code == 401

    login = cliente.post("/api/auth/login", json={"correo": "ana@correo.com", "password": "nueva-clave-123"}).json()
    assert cliente.get("/api/intentos/mios", headers={"Authorization": f"Bearer {login['token']}"}).status_code == 200


def test_validaciones(cliente, registrar, buzon):
    registrar("ana", "estudiante")
    pedir(cliente)
    corta = cliente.post("/api/auth/restablecer", json={"token": token_de(buzon[0]), "password_nueva": "corta"})
    assert corta.status_code == 422
    assert cliente.post("/api/auth/restablecer/verificar", json={"token": "x" * 40}).status_code == 400


def test_sin_smtp_se_escribe_en_la_consola(monkeypatch, capsys):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    assert correo.enviar("ana@correo.com", "Asunto", "enlace: http://x") is False
    assert "http://x" in capsys.readouterr().out
