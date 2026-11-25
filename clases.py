"""Clases (grupos): el docente crea una clase, comparte su código y le asigna cuestionarios.

Un estudiante solo ve y puede entregar los cuestionarios publicados que estén
asignados a alguna de sus clases (ver `cuestionarios_visibles`).
"""
import secrets

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth import usuario_actual, docente_actual, cuestionario_propio
from database import Clase, ClaseEstudiante, ClaseCuestionario, Cuestionario, Usuario, get_db

router = APIRouter(prefix="/api/clases", tags=["clases"])

# Sin caracteres que se confunden al dictarlos o copiarlos (0/O, 1/I/L)
ALFABETO_CODIGO = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
LARGO_CODIGO = 6


class NuevaClase(BaseModel):
    nombre: str = Field(min_length=1, max_length=100)
    descripcion: str | None = Field(default=None, max_length=300)


class Unirse(BaseModel):
    codigo: str = Field(min_length=1, max_length=20)


# ---------- utilidades (también las usa main.py) ----------

def generar_codigo(db: Session) -> str:
    while True:
        codigo = "".join(secrets.choice(ALFABETO_CODIGO) for _ in range(LARGO_CODIGO))
        if not db.query(Clase.id).filter(Clase.codigo == codigo).first():
            return codigo


def ids_clases_del_estudiante(estudiante_id: int, db: Session) -> list[int]:
    return [fila.clase_id for fila in db.query(ClaseEstudiante.clase_id).filter(ClaseEstudiante.estudiante_id == estudiante_id)]


def cuestionarios_visibles(estudiante_id: int, db: Session) -> list[tuple[Cuestionario, list[str]]]:
    """Cuestionarios publicados asignados a las clases del estudiante, con los nombres de esas clases."""
    filas = db.query(Cuestionario, Clase.nombre) \
        .join(ClaseCuestionario, ClaseCuestionario.cuestionario_id == Cuestionario.id) \
        .join(Clase, Clase.id == ClaseCuestionario.clase_id) \
        .join(ClaseEstudiante, ClaseEstudiante.clase_id == Clase.id) \
        .filter(ClaseEstudiante.estudiante_id == estudiante_id, Cuestionario.publicado == True) \
        .order_by(Cuestionario.id.desc()).all()
    por_id: dict[int, tuple[Cuestionario, list[str]]] = {}
    for cuestionario, nombre_clase in filas:
        por_id.setdefault(cuestionario.id, (cuestionario, []))[1].append(nombre_clase)
    return list(por_id.values())


def estudiante_puede_ver(estudiante_id: int, cuestionario_id: int, db: Session) -> bool:
    return db.query(ClaseCuestionario.id) \
        .join(ClaseEstudiante, ClaseEstudiante.clase_id == ClaseCuestionario.clase_id) \
        .filter(ClaseEstudiante.estudiante_id == estudiante_id, ClaseCuestionario.cuestionario_id == cuestionario_id) \
        .first() is not None


def borrar_asignaciones_de_cuestionarios(cuestionario_ids: list[int], db: Session):
    db.query(ClaseCuestionario).filter(ClaseCuestionario.cuestionario_id.in_(cuestionario_ids)) \
        .delete(synchronize_session=False)


def borrar_clases_de_docente(docente_id: int, db: Session):
    """Clases del docente con sus membresías y asignaciones (sin hacer commit)."""
    ids = [fila.id for fila in db.query(Clase.id).filter(Clase.docente_id == docente_id)]
    if not ids:
        return
    db.query(ClaseEstudiante).filter(ClaseEstudiante.clase_id.in_(ids)).delete(synchronize_session=False)
    db.query(ClaseCuestionario).filter(ClaseCuestionario.clase_id.in_(ids)).delete(synchronize_session=False)
    db.query(Clase).filter(Clase.id.in_(ids)).delete(synchronize_session=False)


def borrar_membresias_de_estudiante(estudiante_id: int, db: Session):
    db.query(ClaseEstudiante).filter(ClaseEstudiante.estudiante_id == estudiante_id).delete(synchronize_session=False)


def clase_propia(clase_id: int, usuario: Usuario, db: Session) -> Clase:
    clase = db.query(Clase).filter(Clase.id == clase_id).first()
    if clase is None:
        raise HTTPException(status_code=404, detail="Clase no encontrada.")
    if clase.docente_id != usuario.id:
        raise HTTPException(status_code=403, detail="Esta clase pertenece a otro docente.")
    return clase


def ids_clases_de_cuestionario(cuestionario_id: int, db: Session) -> list[int]:
    filas = db.query(ClaseCuestionario.clase_id).filter(ClaseCuestionario.cuestionario_id == cuestionario_id)
    return sorted(fila.clase_id for fila in filas)


def sincronizar_asignaciones(cuestionario: Cuestionario, clase_ids: list[int], usuario: Usuario, db: Session):
    """
    Deja el cuestionario asignado exactamente a `clase_ids` (todas del docente; si no, 404/403).
    No hace commit: lo hace quien llama, para que publicar y asignar sean una sola operación.
    """
    deseadas = set(clase_ids)
    for clase_id in deseadas:
        clase_propia(clase_id, usuario, db)
    actuales = set(ids_clases_de_cuestionario(cuestionario.id, db))
    sobrantes = actuales - deseadas
    if sobrantes:
        db.query(ClaseCuestionario).filter(
            ClaseCuestionario.cuestionario_id == cuestionario.id, ClaseCuestionario.clase_id.in_(sobrantes)
        ).delete(synchronize_session=False)
    for clase_id in sorted(deseadas - actuales):
        db.add(ClaseCuestionario(clase_id=clase_id, cuestionario_id=cuestionario.id))


def _contar(modelo, clase_id: int, db: Session) -> int:
    return db.query(modelo).filter(modelo.clase_id == clase_id).count()


def clase_a_dict(clase: Clase, db: Session) -> dict:
    return {
        "id": clase.id,
        "nombre": clase.nombre,
        "descripcion": clase.descripcion,
        "codigo": clase.codigo,
        "fecha_creacion": clase.fecha_creacion,
        "num_estudiantes": _contar(ClaseEstudiante, clase.id, db),
        "num_cuestionarios": _contar(ClaseCuestionario, clase.id, db),
    }


# ---------- endpoints ----------

@router.get("")
def listar_clases(usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    """Docente: sus clases. Estudiante: las clases a las que se unió (con el nombre del docente)."""
    if usuario.rol == "docente":
        clases = db.query(Clase).filter(Clase.docente_id == usuario.id).order_by(Clase.id.desc()).all()
        return {"data": [clase_a_dict(c, db) for c in clases]}

    filas = db.query(Clase, Usuario.nombre) \
        .join(ClaseEstudiante, ClaseEstudiante.clase_id == Clase.id) \
        .join(Usuario, Usuario.id == Clase.docente_id) \
        .filter(ClaseEstudiante.estudiante_id == usuario.id) \
        .order_by(ClaseEstudiante.fecha_union.desc()).all()
    return {"data": [
        {"id": c.id, "nombre": c.nombre, "descripcion": c.descripcion, "docente": nombre_docente}
        for c, nombre_docente in filas
    ]}


@router.post("")
def crear_clase(datos: NuevaClase, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    nombre = datos.nombre.strip()
    if not nombre:
        raise HTTPException(status_code=400, detail="La clase necesita un nombre.")
    clase = Clase(
        nombre=nombre,
        descripcion=(datos.descripcion or "").strip() or None,
        codigo=generar_codigo(db),
        docente_id=usuario.id,
    )
    db.add(clase)
    db.commit()
    db.refresh(clase)
    return {"clase": clase_a_dict(clase, db)}


@router.get("/{clase_id}")
def detalle_clase(clase_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    """Docente: la clase con sus estudiantes y los cuestionarios asignados."""
    clase = clase_propia(clase_id, usuario, db)
    estudiantes = db.query(Usuario.id, Usuario.nombre, Usuario.correo, ClaseEstudiante.fecha_union) \
        .join(ClaseEstudiante, ClaseEstudiante.estudiante_id == Usuario.id) \
        .filter(ClaseEstudiante.clase_id == clase.id) \
        .order_by(Usuario.nombre).all()
    cuestionarios = db.query(Cuestionario.id, Cuestionario.nombre_examen, Cuestionario.materia, Cuestionario.publicado) \
        .join(ClaseCuestionario, ClaseCuestionario.cuestionario_id == Cuestionario.id) \
        .filter(ClaseCuestionario.clase_id == clase.id) \
        .order_by(ClaseCuestionario.fecha_asignacion.desc()).all()
    return {
        "clase": clase_a_dict(clase, db),
        "estudiantes": [
            {"id": e.id, "nombre": e.nombre, "correo": e.correo, "fecha_union": e.fecha_union}
            for e in estudiantes
        ],
        "cuestionarios": [
            {"id": c.id, "nombre_examen": c.nombre_examen, "materia": c.materia, "publicado": c.publicado}
            for c in cuestionarios
        ],
    }


@router.put("/{clase_id}")
def editar_clase(clase_id: int, datos: NuevaClase, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    """Cambia el nombre y la descripción (el código no cambia)."""
    clase = clase_propia(clase_id, usuario, db)
    nombre = datos.nombre.strip()
    if not nombre:
        raise HTTPException(status_code=400, detail="La clase necesita un nombre.")
    clase.nombre = nombre
    clase.descripcion = (datos.descripcion or "").strip() or None
    db.commit()
    return {"clase": clase_a_dict(clase, db)}


def clases_de_cuestionario(cuestionario_id: int, db: Session) -> tuple[list[dict], dict[int, list[int]]]:
    """
    Para la vista de resultados: las clases a las que está asignado el cuestionario y,
    por estudiante, en cuáles de esas clases está inscrito actualmente.
    """
    asignadas = db.query(Clase.id, Clase.nombre) \
        .join(ClaseCuestionario, ClaseCuestionario.clase_id == Clase.id) \
        .filter(ClaseCuestionario.cuestionario_id == cuestionario_id) \
        .order_by(Clase.nombre).all()
    ids = [c.id for c in asignadas]
    por_estudiante: dict[int, list[int]] = {}
    if ids:
        for fila in db.query(ClaseEstudiante.estudiante_id, ClaseEstudiante.clase_id).filter(ClaseEstudiante.clase_id.in_(ids)):
            por_estudiante.setdefault(fila.estudiante_id, []).append(fila.clase_id)
    return [{"id": c.id, "nombre": c.nombre} for c in asignadas], por_estudiante


@router.delete("/{clase_id}")
def eliminar_clase(clase_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    """Borra la clase y sus membresías/asignaciones. Los cuestionarios y los intentos se conservan."""
    clase = clase_propia(clase_id, usuario, db)
    db.query(ClaseEstudiante).filter(ClaseEstudiante.clase_id == clase.id).delete()
    db.query(ClaseCuestionario).filter(ClaseCuestionario.clase_id == clase.id).delete()
    db.delete(clase)
    db.commit()
    return {"mensaje": "Clase eliminada"}


@router.put("/{clase_id}/codigo")
def regenerar_codigo(clase_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    """Nuevo código de acceso (el anterior deja de funcionar; los estudiantes ya unidos se quedan)."""
    clase = clase_propia(clase_id, usuario, db)
    clase.codigo = generar_codigo(db)
    db.commit()
    return {"codigo": clase.codigo}


@router.post("/{clase_id}/cuestionarios/{cuestionario_id}")
def asignar_cuestionario(clase_id: int, cuestionario_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    clase = clase_propia(clase_id, usuario, db)
    cuestionario = cuestionario_propio(cuestionario_id, usuario, db)
    existe = db.query(ClaseCuestionario.id).filter(
        ClaseCuestionario.clase_id == clase.id, ClaseCuestionario.cuestionario_id == cuestionario.id
    ).first()
    if not existe:
        db.add(ClaseCuestionario(clase_id=clase.id, cuestionario_id=cuestionario.id))
        try:
            db.commit()
        except IntegrityError:  # doble clic: ya quedó asignado
            db.rollback()
    return {"mensaje": "Cuestionario asignado a la clase"}


@router.delete("/{clase_id}/cuestionarios/{cuestionario_id}")
def quitar_cuestionario(clase_id: int, cuestionario_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    clase = clase_propia(clase_id, usuario, db)
    db.query(ClaseCuestionario).filter(
        ClaseCuestionario.clase_id == clase.id, ClaseCuestionario.cuestionario_id == cuestionario_id
    ).delete()
    db.commit()
    return {"mensaje": "Cuestionario quitado de la clase"}


@router.delete("/{clase_id}/estudiantes/{estudiante_id}")
def quitar_estudiante(clase_id: int, estudiante_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    clase = clase_propia(clase_id, usuario, db)
    db.query(ClaseEstudiante).filter(
        ClaseEstudiante.clase_id == clase.id, ClaseEstudiante.estudiante_id == estudiante_id
    ).delete()
    db.commit()
    return {"mensaje": "Estudiante quitado de la clase"}


@router.post("/unirse")
def unirse_a_clase(datos: Unirse, usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    """Estudiante: se une a una clase con su código (no distingue mayúsculas, ignora espacios y guiones)."""
    if usuario.rol != "estudiante":
        raise HTTPException(status_code=403, detail="Solo los estudiantes pueden unirse a una clase.")
    codigo = "".join(ch for ch in datos.codigo.upper() if ch.isalnum())
    clase = db.query(Clase).filter(Clase.codigo == codigo).first()
    if clase is None:
        raise HTTPException(status_code=404, detail="No existe ninguna clase con ese código. Revísalo con tu docente.")

    ya_inscrito = db.query(ClaseEstudiante.id).filter(
        ClaseEstudiante.clase_id == clase.id, ClaseEstudiante.estudiante_id == usuario.id
    ).first()
    if ya_inscrito:
        raise HTTPException(status_code=409, detail=f"Ya perteneces a la clase «{clase.nombre}».")

    db.add(ClaseEstudiante(clase_id=clase.id, estudiante_id=usuario.id))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail=f"Ya perteneces a la clase «{clase.nombre}».")
    return {"mensaje": f"Te uniste a «{clase.nombre}»", "clase": {"id": clase.id, "nombre": clase.nombre}}


@router.delete("/{clase_id}/salir")
def salir_de_clase(clase_id: int, usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    """Estudiante: abandona una clase."""
    db.query(ClaseEstudiante).filter(
        ClaseEstudiante.clase_id == clase_id, ClaseEstudiante.estudiante_id == usuario.id
    ).delete()
    db.commit()
    return {"mensaje": "Saliste de la clase"}
