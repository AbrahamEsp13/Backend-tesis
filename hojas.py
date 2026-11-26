"""4.10 Calificar con foto: endpoints.

Flujo del docente (desde Resultados de un examen):
  1. GET  .../hojas/estudiantes  -> los estudiantes inscritos en las clases a las que está asignado.
  2. POST .../hojas/leer         -> sube la foto; devuelve lo leído y una imagen de control. No guarda nada.
  3. Revisa y corrige lo leído, elige al estudiante.
  4. POST .../hojas/registrar    -> crea el intento (marcado `en_papel`) calificado con la clave actual.

Solo estudiantes con cuenta e inscritos (el intento aparece en sus resultados, con retroalimentación).
Un intento en papel cuenta como un intento más (también para los intentos máximos en línea).
"""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

import clases
from auth import docente_actual, cuestionario_propio
from calificacion import calificar
from database import Clase, ClaseCuestionario, ClaseEstudiante, Intento, Usuario, get_db
from lectura_hojas import HojaIlegible, disposicion, leer_hoja

router = APIRouter(prefix="/api/cuestionarios/{cuestionario_id}/hojas", tags=["hojas"])

MAX_FOTO = 15 * 1024 * 1024  # 15 MB


class Registro(BaseModel):
    estudiante_id: int
    # { "<índice de pregunta>": índice de la opción elegida, o None si la dejó en blanco }
    respuestas: dict[str, int | None] = Field(default_factory=dict)


def opciones_por_pregunta(preguntas: list) -> list[int]:
    return [len(p.get("opciones") or []) for p in preguntas]


@router.get("/estudiantes")
def estudiantes(cuestionario_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    cuestionario = cuestionario_propio(cuestionario_id, usuario, db)
    filas = db.query(Usuario.id, Usuario.nombre, Usuario.correo, Clase.nombre) \
        .join(ClaseEstudiante, ClaseEstudiante.estudiante_id == Usuario.id) \
        .join(Clase, Clase.id == ClaseEstudiante.clase_id) \
        .join(ClaseCuestionario, ClaseCuestionario.clase_id == Clase.id) \
        .filter(ClaseCuestionario.cuestionario_id == cuestionario.id) \
        .order_by(Usuario.nombre).all()
    en_papel = dict(db.query(Intento.estudiante_id, func.count(Intento.id)).filter(
        Intento.cuestionario_id == cuestionario.id, Intento.en_papel == True  # noqa: E712
    ).group_by(Intento.estudiante_id).all())
    por_id: dict[int, dict] = {}
    for estudiante_id, nombre, correo, clase in filas:
        registro = por_id.setdefault(estudiante_id, {
            "id": estudiante_id, "nombre": nombre, "correo": correo, "clases": [],
            "intentos_en_papel": en_papel.get(estudiante_id, 0),
        })
        if clase not in registro["clases"]:
            registro["clases"].append(clase)
    return {
        "estudiantes": list(por_id.values()),
        "se_puede_leer": disposicion(opciones_por_pregunta(cuestionario.preguntas_json or [])) is not None,
    }


# "def": el procesamiento de la imagen ocupa la CPU un momento; FastAPI lo corre en otro hilo.
@router.post("/leer")
def leer(cuestionario_id: int, foto: UploadFile = File(...), usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    cuestionario = cuestionario_propio(cuestionario_id, usuario, db)
    contenido = foto.file.read(MAX_FOTO + 1)
    if len(contenido) > MAX_FOTO:
        raise HTTPException(status_code=413, detail="La foto supera los 15 MB.")
    preguntas = cuestionario.preguntas_json or []
    try:
        resultado = leer_hoja(contenido, opciones_por_pregunta(preguntas))
    except HojaIlegible as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print(f"ERROR al leer la hoja: {e}")
        raise HTTPException(status_code=422, detail="No se pudo procesar la foto. Intenta con otra.")

    lectura = [
        {**r, "pregunta": i, "opcion": preguntas[i]["opciones"][r["indice"]] if r["indice"] is not None else None}
        for i, r in enumerate(resultado["lectura"])
    ]
    revisar = sum(r["estado"] != "ok" for r in lectura)
    print(f"INFO hoja leída: {len(lectura)} preguntas, {revisar} por revisar")
    return {"lectura": lectura, "vista": resultado["vista"], "por_revisar": revisar}


@router.post("/registrar")
def registrar(cuestionario_id: int, datos: Registro, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    from main import intento_a_dict, preguntas_con_ids  # aquí: main importa este módulo
    cuestionario = cuestionario_propio(cuestionario_id, usuario, db)
    estudiante = db.query(Usuario).filter(Usuario.id == datos.estudiante_id, Usuario.rol == "estudiante").first()
    if estudiante is None or not clases.estudiante_puede_ver(estudiante.id, cuestionario.id, db):
        raise HTTPException(status_code=400, detail="Ese estudiante no está inscrito en una clase con este examen.")

    preguntas = preguntas_con_ids(cuestionario, db)
    respuestas = {}
    for clave, indice in datos.respuestas.items():
        if indice is None or not clave.isdigit() or int(clave) >= len(preguntas):
            continue
        ids = preguntas[int(clave)].get("ids_opciones") or []
        if not 0 <= indice < len(ids):
            raise HTTPException(status_code=400, detail=f"La pregunta {int(clave) + 1} no tiene la opción {indice + 1}.")
        respuestas[clave] = ids[indice]
    resultado = calificar(preguntas, respuestas)
    intento = Intento(
        cuestionario_id=cuestionario.id,
        estudiante_id=estudiante.id,
        respuestas=respuestas,
        puntos_obtenidos=resultado["puntos_obtenidos"],
        puntos_totales=resultado["puntos_totales"],
        calificacion=resultado["calificacion"],
        tiempo_segundos=None,
        seguridad={},
        en_papel=True,
    )
    db.add(intento)
    db.commit()
    db.refresh(intento)
    return {"intento": intento_a_dict(intento, preguntas), "nombre_estudiante": estudiante.nombre}
