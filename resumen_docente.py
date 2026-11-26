"""Inicio del docente (3.2): lo que conviene atender, en una sola petición.

- entregas_recientes: los últimos intentos en sus exámenes.
- borradores: exámenes sin publicar.
- publicados_sin_clase: publicados pero sin asignar a ninguna clase (nadie los ve).
- clases: con número de estudiantes y exámenes, y el promedio de la clase
  (mejor intento de cada estudiante inscrito en cada examen asignado a la clase).
- reportes: preguntas reportadas por los estudiantes, por examen.
"""
from collections import defaultdict

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from auth import docente_actual
import configuracion
from database import Clase, ClaseCuestionario, ClaseEstudiante, Cuestionario, Intento, Reporte, Usuario, get_db

router = APIRouter(tags=["inicio"])

MAX_ENTREGAS = 8


def _nombre(cuestionario) -> str:
    return cuestionario.nombre_examen or cuestionario.nombre_documento or "Examen sin título"


def _promedios_por_clase(clases: list, db: Session) -> dict[int, dict]:
    ids = [c.id for c in clases]
    if not ids:
        return {}
    miembros = defaultdict(set)
    for fila in db.query(ClaseEstudiante.clase_id, ClaseEstudiante.estudiante_id).filter(ClaseEstudiante.clase_id.in_(ids)):
        miembros[fila.clase_id].add(fila.estudiante_id)
    examenes = defaultdict(set)
    for fila in db.query(ClaseCuestionario.clase_id, ClaseCuestionario.cuestionario_id).filter(ClaseCuestionario.clase_id.in_(ids)):
        examenes[fila.clase_id].add(fila.cuestionario_id)

    todos_estudiantes = set().union(*miembros.values()) if miembros else set()
    todos_examenes = set().union(*examenes.values()) if examenes else set()
    mejores: dict[tuple[int, int], float] = {}
    if todos_estudiantes and todos_examenes:
        filas = db.query(Intento.estudiante_id, Intento.cuestionario_id, func.max(Intento.calificacion).label("mejor")) \
            .filter(Intento.estudiante_id.in_(todos_estudiantes), Intento.cuestionario_id.in_(todos_examenes)) \
            .group_by(Intento.estudiante_id, Intento.cuestionario_id)
        mejores = {(f.estudiante_id, f.cuestionario_id): f.mejor for f in filas}

    resultado = {}
    for clase_id in ids:
        notas = [
            nota for (estudiante, examen), nota in mejores.items()
            if estudiante in miembros[clase_id] and examen in examenes[clase_id]
        ]
        resultado[clase_id] = {
            "num_estudiantes": len(miembros[clase_id]),
            "num_cuestionarios": len(examenes[clase_id]),
            "entregas": len(notas),
            "promedio": round(sum(notas) / len(notas), 2) if notas else None,
        }
    return resultado


@router.get("/api/docente/resumen")
def resumen_docente(usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    cuestionarios = db.query(Cuestionario).filter(Cuestionario.usuario_id == usuario.id) \
        .order_by(Cuestionario.id.desc()).all()
    por_id = {c.id: c for c in cuestionarios}
    ids = list(por_id)
    configuracion.entregar_vencidos(db, cuestionario_ids=ids)  # que aparezcan las entregas automáticas

    entregas = []
    asignados: set[int] = set()
    reportes = []
    if ids:
        filas = db.query(Intento, Usuario.nombre) \
            .join(Usuario, Usuario.id == Intento.estudiante_id) \
            .filter(Intento.cuestionario_id.in_(ids)) \
            .order_by(Intento.fecha.desc(), Intento.id.desc()).limit(MAX_ENTREGAS).all()
        entregas = [
            {
                "intento_id": intento.id,
                "cuestionario_id": intento.cuestionario_id,
                "nombre_examen": _nombre(por_id[intento.cuestionario_id]),
                "nombre_estudiante": nombre,
                "calificacion": intento.calificacion,
                "fecha": intento.fecha,
            }
            for intento, nombre in filas
        ]
        asignados = {f.cuestionario_id for f in db.query(ClaseCuestionario.cuestionario_id).filter(ClaseCuestionario.cuestionario_id.in_(ids))}
        conteo = db.query(Reporte.cuestionario_id, func.count(Reporte.id).label("n")) \
            .filter(Reporte.cuestionario_id.in_(ids)).group_by(Reporte.cuestionario_id).all()
        reportes = sorted(
            ({"cuestionario_id": f.cuestionario_id, "nombre_examen": _nombre(por_id[f.cuestionario_id]), "reportes": f.n} for f in conteo),
            key=lambda r: -r["reportes"],
        )

    def examen_corto(c):
        return {"id": c.id, "nombre_examen": _nombre(c), "materia": c.materia,
                "fecha_creacion": c.fecha_creacion, "num_preguntas": len(c.preguntas_json or [])}

    clases = db.query(Clase).filter(Clase.docente_id == usuario.id).order_by(Clase.nombre).all()
    datos_clases = _promedios_por_clase(clases, db)

    return {
        "entregas_recientes": entregas,
        "borradores": [examen_corto(c) for c in cuestionarios if not c.publicado],
        "publicados_sin_clase": [examen_corto(c) for c in cuestionarios if c.publicado and c.id not in asignados],
        "clases": [{"id": c.id, "nombre": c.nombre, "codigo": c.codigo, **datos_clases[c.id]} for c in clases],
        "reportes": reportes,
        "total_examenes": len(cuestionarios),
    }
