"""Calidad del generador (2.7): el docente valora cada pregunta generada como útil o no útil.

- La valoración se guarda en la propia pregunta: `valoracion: {util, motivo, fecha}`.
  Se puede valorar aunque el examen esté publicado (no cambia lo que ve el estudiante).
- Las preguntas agregadas a mano (`origen: "manual"`) no cuentan en las estadísticas.
- `estadisticas(cuestionarios)` agrega por nivel de complejidad; la usan el endpoint
  (exámenes del docente) y el script exportar_valoraciones.py (todos, para la tesis).
"""
import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import docente_actual, cuestionario_propio
from database import Cuestionario, Usuario, get_db

router = APIRouter(tags=["valoraciones"])

MOTIVOS = ("ambigua", "fuera_del_tema", "clave_incorrecta", "demasiado_facil", "otro")
Motivo = Literal["ambigua", "fuera_del_tema", "clave_incorrecta", "demasiado_facil", "otro"]
SIN_COMPLEJIDAD = "sin_dato"  # exámenes generados antes de que existiera la complejidad


class Valoracion(BaseModel):
    util: bool | None  # None quita la valoración
    motivo: Motivo | None = None  # solo cuando no es útil


def es_generada(pregunta: dict) -> bool:
    return pregunta.get("origen") != "manual"


def sin_valoraciones(preguntas: list) -> list:
    """Copia de las preguntas sin valoración (al clonar: la valoración es de quien la hizo)."""
    return [{k: v for k, v in p.items() if k != "valoracion"} for p in preguntas or []]


def _vacio() -> dict:
    return {
        "generadas": 0, "con_fuente": 0, "con_cita_literal": 0,
        "valoradas": 0, "utiles": 0, "no_utiles": 0,
        "motivos": {m: 0 for m in MOTIVOS},
    }


def estadisticas(cuestionarios) -> dict:
    """{"total": {...}, "por_complejidad": {"basico": {...}, ...}} sobre las preguntas generadas."""
    total = _vacio()
    por_complejidad: dict[str, dict] = {}
    for cuestionario in cuestionarios:
        grupo = por_complejidad.setdefault(cuestionario.complejidad or SIN_COMPLEJIDAD, _vacio())
        for pregunta in cuestionario.preguntas_json or []:
            if not isinstance(pregunta, dict) or not es_generada(pregunta):
                continue
            fuente = pregunta.get("fuente") or {}
            valoracion = pregunta.get("valoracion") or {}
            for destino in (total, grupo):
                destino["generadas"] += 1
                destino["con_fuente"] += bool(fuente)
                destino["con_cita_literal"] += bool(fuente.get("literal"))
                if valoracion.get("util") is True:
                    destino["valoradas"] += 1
                    destino["utiles"] += 1
                elif valoracion.get("util") is False:
                    destino["valoradas"] += 1
                    destino["no_utiles"] += 1
                    if valoracion.get("motivo") in destino["motivos"]:
                        destino["motivos"][valoracion["motivo"]] += 1
    return {"total": total, "por_complejidad": por_complejidad}


@router.put("/api/cuestionarios/{cuestionario_id}/preguntas/{indice}/valoracion")
def valorar_pregunta(cuestionario_id: int, indice: int, datos: Valoracion, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    cuestionario = cuestionario_propio(cuestionario_id, usuario, db)
    preguntas = [dict(p) for p in cuestionario.preguntas_json or []]
    if not 0 <= indice < len(preguntas):
        raise HTTPException(status_code=404, detail="Esa pregunta no existe en el examen.")
    if datos.util is None:
        preguntas[indice].pop("valoracion", None)
        valoracion = None
    else:
        valoracion = {
            "util": datos.util,
            "motivo": None if datos.util else datos.motivo,
            "fecha": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        preguntas[indice]["valoracion"] = valoracion
    cuestionario.preguntas_json = preguntas  # lista nueva: SQLAlchemy detecta el cambio del JSON
    db.commit()
    return {"valoracion": valoracion}


@router.get("/api/estadisticas/generador")
def estadisticas_del_docente(usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    """Calidad de las preguntas generadas en los exámenes del docente."""
    cuestionarios = db.query(Cuestionario).filter(Cuestionario.usuario_id == usuario.id).all()
    return estadisticas(cuestionarios)
