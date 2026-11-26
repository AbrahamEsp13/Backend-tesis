"""4.5 Formatos de impresión: CRUD personal del docente, logos validados y un solo predeterminado."""
import pytest

import formatos

LOGO = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


@pytest.fixture
def docente(registrar):
    return registrar("docente", "docente")[0]


def crear(cliente, docente, **cambios):
    cuerpo = {"nombre": "Secundaria 12", "datos": {"escuela": "Escuela Secundaria Técnica 12", "lineas": ["CCT 09DST0012X", "Turno matutino"]},
              "logo_izquierdo": LOGO, **cambios}
    return cliente.post("/api/formatos", headers=docente, json=cuerpo)


def test_crear_y_listar(cliente, docente):
    respuesta = crear(cliente, docente)
    assert respuesta.status_code == 200, respuesta.text
    formato = respuesta.json()["formato"]
    assert formato["datos"]["escuela"] == "Escuela Secundaria Técnica 12"
    assert formato["datos"]["campos"] == ["nombre", "grupo", "fecha", "calificacion"]  # predeterminados
    assert formato["datos"]["mostrar"]["titulo"] is True
    assert formato["logo_izquierdo"] == LOGO and formato["logo_derecho"] is None
    assert [f["id"] for f in cliente.get("/api/formatos", headers=docente).json()["formatos"]] == [formato["id"]]


def test_validaciones(cliente, docente):
    assert crear(cliente, docente, nombre="  ").status_code == 422
    assert crear(cliente, docente, logo_derecho="data:text/html;base64,PHNjcmlwdD4=").status_code == 422
    assert crear(cliente, docente, logo_derecho="data:image/png;base64," + "A" * formatos.MAX_LOGO).status_code == 422
    assert crear(cliente, docente, datos={"lineas": ["1", "2", "3", "4"]}).status_code == 422
    assert crear(cliente, docente, datos={"campos": ["nombre", "contrasena"]}).status_code == 422
    repetidos = crear(cliente, docente, datos={"campos": ["nombre", "nombre", "grupo"]}).json()["formato"]
    assert repetidos["datos"]["campos"] == ["nombre", "grupo"]


def test_un_solo_predeterminado(cliente, docente):
    a = crear(cliente, docente, nombre="A", predeterminado=True).json()["formato"]
    b = crear(cliente, docente, nombre="B", predeterminado=True).json()["formato"]
    lista = cliente.get("/api/formatos", headers=docente).json()["formatos"]
    assert [(f["id"], f["predeterminado"]) for f in lista] == [(b["id"], True), (a["id"], False)]

    cliente.put(f"/api/formatos/{a['id']}", headers=docente, json={"nombre": "A", "predeterminado": True})
    lista = cliente.get("/api/formatos", headers=docente).json()["formatos"]
    assert {f["nombre"]: f["predeterminado"] for f in lista} == {"A": True, "B": False}


def test_actualizar_duplicar_y_eliminar(cliente, docente):
    formato = crear(cliente, docente, predeterminado=True).json()["formato"]
    editado = cliente.put(f"/api/formatos/{formato['id']}", headers=docente,
                          json={"nombre": "Nuevo", "datos": {"escuela": "Otra"}, "logo_derecho": LOGO, "predeterminado": True}).json()["formato"]
    assert editado["nombre"] == "Nuevo" and editado["logo_izquierdo"] is None and editado["logo_derecho"] == LOGO

    copia = cliente.post(f"/api/formatos/{formato['id']}/duplicar", headers=docente).json()["formato"]
    assert copia["nombre"] == "Nuevo (copia)" and copia["predeterminado"] is False and copia["logo_derecho"] == LOGO

    assert cliente.delete(f"/api/formatos/{formato['id']}", headers=docente).status_code == 200
    assert [f["id"] for f in cliente.get("/api/formatos", headers=docente).json()["formatos"]] == [copia["id"]]


def test_son_personales(cliente, docente, registrar):
    formato = crear(cliente, docente).json()["formato"]
    otro, _ = registrar("otra", "docente")
    estudiante, _ = registrar("ana", "estudiante")
    assert cliente.get("/api/formatos", headers=otro).json()["formatos"] == []
    assert cliente.put(f"/api/formatos/{formato['id']}", headers=otro, json={"nombre": "x"}).status_code == 404
    assert cliente.delete(f"/api/formatos/{formato['id']}", headers=otro).status_code == 404
    assert cliente.post(f"/api/formatos/{formato['id']}/duplicar", headers=otro).status_code == 404
    assert cliente.get("/api/formatos", headers=estudiante).status_code == 403


def test_limite_de_formatos(cliente, docente, monkeypatch):
    monkeypatch.setattr(formatos, "MAX_FORMATOS", 2)
    primero = crear(cliente, docente).json()["formato"]
    crear(cliente, docente)
    assert crear(cliente, docente).status_code == 400
    assert cliente.post(f"/api/formatos/{primero['id']}/duplicar", headers=docente).status_code == 400


def test_borrar_la_cuenta_borra_sus_formatos(cliente, docente, db):
    from database import FormatoImpresion, Usuario
    crear(cliente, docente)
    yo = db.query(Usuario).filter(Usuario.rol == "docente").first()
    respuesta = cliente.request("DELETE", f"/api/usuarios/{yo.id}", headers=docente, json={"password": "secreta123"})
    assert respuesta.status_code == 200, respuesta.text
    db.expire_all()
    assert db.query(FormatoImpresion).count() == 0
