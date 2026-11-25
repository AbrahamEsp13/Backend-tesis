"""Reportes de preguntas: el estudiante avisa que una pregunta parece incorrecta o ambigua.

- Se reporta desde la revisión de un intento propio ya entregado.
- Un reporte por estudiante, examen y pregunta; reportar de nuevo actualiza motivo y comentario.
- El docente los ve en Resultados (junto con los intentos) y puede descartarlos.
"""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth import usuario_actual, docente_actual, cuestionario_propio
from database import Cuestionario, Intento, Reporte, Usuario, get_db

router = APIRouter(tags=["reportes"])

Motivo = Literal["clave_incorrecta", "ambigua", "fuera_del_tema", "otro"]


class NuevoReporte(BaseModel):
    pregunta: int = Field(ge=0)
    motivo: Motivo
    comentario: str | None = Field(default=None, max_length=500)


def reporte_a_dict(reporte: Reporte, nombre_estudiante: str | None = None) -> dict:
    return {
        "id": reporte.id,
        "pregunta": reporte.pregunta,
        "texto_pregunta": reporte.texto_pregunta,
        "motivo": reporte.motivo,
        "comentario": reporte.comentario,
        "fecha": reporte.fecha,
        "nombre_estudiante": nombre_estudiante,
    }


def reportes_de_cuestionario(cuestionario_id: int, db: Session) -> list[dict]:
    """Para el docente: todos los reportes del examen, del más reciente al más antiguo."""
    filas = db.query(Reporte, Usuario.nombre) \
        .join(Usuario, Usuario.id == Reporte.estudiante_id) \
        .filter(Reporte.cuestionario_id == cuestionario_id) \
        .order_by(Reporte.fecha.desc()).all()
    return [reporte_a_dict(r, nombre) for r, nombre in filas]


def preguntas_reportadas(estudiante_id: int, cuestionario_id: int, db: Session) -> list[int]:
    filas = db.query(Reporte.pregunta).filter(
        Reporte.estudiante_id == estudiante_id, Reporte.cuestionario_id == cuestionario_id
    )
    return sorted(f.pregunta for f in filas)


def borrar_reportes(db: Session, *, cuestionario_ids: list[int] | None = None, estudiante_id: int | None = None):
    consulta = db.query(Reporte)
    if cuestionario_ids is not None:
        consulta = consulta.filter(Reporte.cuestionario_id.in_(cuestionario_ids))
    if estudiante_id is not None:
        consulta = consulta.filter(Reporte.estudiante_id == estudiante_id)
    consulta.delete(synchronize_session=False)


@router.post("/api/intentos/{intento_id}/reportes")
def reportar_pregunta(intento_id: int, datos: NuevoReporte, usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    intento = db.query(Intento).filter(Intento.id == intento_id).first()
    if intento is None or intento.estudiante_id != usuario.id:
        raise HTTPException(status_code=404, detail="Intento no encontrado.")
    cuestionario = db.query(Cuestionario).filter(Cuestionario.id == intento.cuestionario_id).first()
    preguntas = (cuestionario.preguntas_json or []) if cuestionario else []
    if datos.pregunta >= len(preguntas):
        raise HTTPException(status_code=400, detail="Esa pregunta no existe en el examen.")

    comentario = (datos.comentario or "").strip() or None
    reporte = db.query(Reporte).filter(
        Reporte.estudiante_id == usuario.id,
        Reporte.cuestionario_id == cuestionario.id,
        Reporte.pregunta == datos.pregunta,
    ).first()
    if reporte is None:
        reporte = Reporte(cuestionario_id=cuestionario.id, estudiante_id=usuario.id, pregunta=datos.pregunta)
        db.add(reporte)
    reporte.intento_id = intento.id
    reporte.texto_pregunta = str(preguntas[datos.pregunta].get("pregunta", ""))
    reporte.motivo = datos.motivo
    reporte.comentario = comentario
    try:
        db.commit()
    except IntegrityError:  # doble clic: el otro envío ya lo creó
        db.rollback()
    return {"mensaje": "Gracias: tu docente revisará la pregunta."}


@router.delete("/api/cuestionarios/{cuestionario_id}/reportes/{reporte_id}")
def descartar_reporte(cuestionario_id: int, reporte_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    cuestionario = cuestionario_propio(cuestionario_id, usuario, db)
    db.query(Reporte).filter(Reporte.id == reporte_id, Reporte.cuestionario_id == cuestionario.id).delete()
    db.commit()
    return {"mensaje": "Reporte descartado"}
