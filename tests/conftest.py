"""Configuración común de las pruebas.

Las variables de entorno se fijan ANTES de importar la aplicación: database.py crea el motor
al importarse y load_dotenv() no sobrescribe variables ya definidas. Así las pruebas usan una
base SQLite temporal y nunca tocan Neon ni llaman a Gemini.
"""
import os
import sys
import tempfile
from pathlib import Path

_DIRECTORIO_TEMPORAL = tempfile.mkdtemp(prefix="quizai-pruebas-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_DIRECTORIO_TEMPORAL, 'pruebas.db').as_posix()}"
os.environ["JWT_SECRET"] = "clave-de-pruebas-con-al-menos-32-caracteres"
os.environ["GEMINI_API_KEY"] = "clave-falsa-de-pruebas"
os.environ["CORS_ORIGINS"] = "http://localhost:5173,https://quizai-pruebas.vercel.app"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from database import Base, Cuestionario, SessionLocal, engine  # noqa: E402

# Salvaguarda: drop_all() se ejecuta en cada prueba; jamás debe apuntar a la base real.
assert engine.url.get_backend_name() == "sqlite", f"Las pruebas deben usar SQLite, no {engine.url!r}"

PASSWORD = "secreta123"


@pytest.fixture(autouse=True)
def base_limpia():
    """Cada prueba empieza con las tablas vacías."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def cliente():
    return TestClient(main.app)


@pytest.fixture
def db():
    sesion = SessionLocal()
    yield sesion
    sesion.close()


@pytest.fixture
def registrar(cliente):
    """registrar("ana", "estudiante") -> (encabezados con el token, usuario del login)."""
    def _registrar(nombre, rol="estudiante", password=PASSWORD):
        correo = f"{nombre}@correo.com"
        respuesta = cliente.post("/api/auth/register", json={"nombre": nombre, "correo": correo, "password": password, "rol": rol})
        assert respuesta.status_code == 200, respuesta.text
        login = cliente.post("/api/auth/login", json={"correo": correo, "password": password}).json()
        return {"Authorization": f"Bearer {login['token']}"}, login["usuario"]
    return _registrar


@pytest.fixture
def crear_cuestionario(db):
    """Inserta un cuestionario directamente (sin pasar por la IA). Devuelve su id."""
    def _crear(docente_id, nombre="Examen", publicado=True, preguntas=None):
        cuestionario = Cuestionario(
            nombre_examen=nombre,
            nombre_documento="material.pdf",
            materia="Programación",
            usuario_id=docente_id,
            publicado=publicado,
            compartido_comunidad=False,
            preguntas_json=preguntas if preguntas is not None else [
                {"pregunta": "p1", "opciones": ["a", "b"], "respuesta_correcta": "a",
                 "justificacion_pedagogica": "j1", "puntuacion": 1},
                {"pregunta": "p2", "opciones": ["a", "b"], "respuesta_correcta": "b",
                 "justificacion_pedagogica": "j2", "puntuacion": 3},
            ],
        )
        db.add(cuestionario)
        db.commit()
        return cuestionario.id
    return _crear


@pytest.fixture
def clase_con_examen(cliente, registrar, crear_cuestionario):
    """Docente con una clase que tiene asignado un examen publicado, y una estudiante inscrita."""
    docente, usuario_docente = registrar("docente", "docente")
    estudiante, usuario_estudiante = registrar("ana", "estudiante")
    examen_id = crear_cuestionario(usuario_docente["id"])
    clase = cliente.post("/api/clases", headers=docente, json={"nombre": "3A"}).json()["clase"]
    cliente.post(f"/api/clases/{clase['id']}/cuestionarios/{examen_id}", headers=docente)
    cliente.post("/api/clases/unirse", headers=estudiante, json={"codigo": clase["codigo"]})
    return {
        "docente": docente, "usuario_docente": usuario_docente,
        "estudiante": estudiante, "usuario_estudiante": usuario_estudiante,
        "examen_id": examen_id, "clase": clase,
    }


@pytest.fixture
def ia_simulada(monkeypatch):
    """Reemplaza el RAG y Gemini; registra la complejidad con la que se llamó la generación."""
    llamadas = []
    fragmentos = [
        {"numero": 1, "texto": "Introducción al curso.", "pagina": 1, "pagina_fin": 1},
        {"numero": 2, "texto": "Una clase define los atributos y métodos de sus objetos.", "pagina": 4, "pagina_fin": 5},
    ]
    monkeypatch.setattr(main, "construir_contexto", lambda contenido: {
        "contexto": "texto del pdf", "fragmentos": fragmentos, "fragmentos_totales": 3, "fragmentos_usados": 2,
    })

    def generar(contexto, num_preguntas, complejidad, tipos=None):
        llamadas.append({"num_preguntas": num_preguntas, "complejidad": complejidad, "tipos": tipos})
        # Como validar_preguntas: con los campos crudos fragmento y cita
        return [{"pregunta": "p", "opciones": ["a) x", "b) y"], "respuesta_correcta": "a) x",
                 "justificacion_pedagogica": "j", "puntuacion": 1,
                 "fragmento": 2, "cita": "define los atributos y métodos"}]
    monkeypatch.setattr(main, "generar_preguntas", generar)
    return llamadas
