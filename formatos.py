"""4.5 Formatos de impresión: el encabezado de hoja que cada docente diseña y reutiliza.

- Cada docente tiene los suyos (no se comparten); puede duplicarlos y marcar uno como predeterminado.
- Un formato tiene hasta dos logos (izquierda y derecha, p. ej. SEP y escuela), el nombre de la
  escuela, hasta 3 líneas más, qué datos del examen mostrar, los campos que llena el alumno,
  instrucciones generales y un pie de página.
- Los logos se guardan como data URL (igual que la foto de perfil); el navegador los reduce antes
  de enviarlos y aquí se limita el tamaño.
- Al imprimir, el docente elige uno de sus formatos o "Hoja blanca"; la impresión se arma en el navegador.
"""
import re
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from auth import docente_actual
from database import FormatoImpresion, Usuario, ahora_utc, get_db

router = APIRouter(prefix="/api/formatos", tags=["formatos"])

MAX_FORMATOS = 20
MAX_LOGO = 400_000  # caracteres del data URL (~300 KB de imagen)
LOGO_VALIDO = re.compile(r"^data:image/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+$")

Campo = Literal["nombre", "grupo", "numero_lista", "fecha", "calificacion", "firma_tutor"]


class MostrarDatos(BaseModel):
    titulo: bool = True
    materia: bool = True
    docente: bool = True
    puntos: bool = True


class DatosFormato(BaseModel):
    escuela: str = Field(default="", max_length=120)
    lineas: list[str] = Field(default_factory=list, max_length=3)
    mostrar: MostrarDatos = Field(default_factory=MostrarDatos)
    campos: list[Campo] = Field(default_factory=lambda: ["nombre", "grupo", "fecha", "calificacion"])
    instrucciones: str = Field(default="", max_length=600)
    pie: str = Field(default="", max_length=150)

    @field_validator("lineas")
    @classmethod
    def _lineas(cls, lineas: list[str]) -> list[str]:
        if any(len(linea) > 120 for linea in lineas):
            raise ValueError("Cada línea del encabezado admite hasta 120 caracteres.")
        return [linea.strip() for linea in lineas]

    @field_validator("campos")
    @classmethod
    def _sin_repetir(cls, campos: list[str]) -> list[str]:
        return list(dict.fromkeys(campos))


class Formato(BaseModel):
    nombre: str = Field(min_length=1, max_length=80)
    datos: DatosFormato = Field(default_factory=DatosFormato)
    logo_izquierdo: str | None = None
    logo_derecho: str | None = None
    predeterminado: bool = False

    @field_validator("nombre")
    @classmethod
    def _nombre(cls, nombre: str) -> str:
        if not nombre.strip():
            raise ValueError("El formato necesita un nombre.")
        return nombre.strip()

    @field_validator("logo_izquierdo", "logo_derecho")
    @classmethod
    def _logo(cls, logo: str | None) -> str | None:
        if not logo:
            return None
        if len(logo) > MAX_LOGO:
            raise ValueError("El logo es demasiado grande (máximo unos 300 KB).")
        if not LOGO_VALIDO.match(logo):
            raise ValueError("El logo debe ser una imagen PNG, JPG o WEBP.")
        return logo


def formato_a_dict(formato: FormatoImpresion) -> dict:
    return {
        "id": formato.id,
        "nombre": formato.nombre,
        "datos": DatosFormato(**(formato.datos or {})).model_dump(),
        "logo_izquierdo": formato.logo_izquierdo,
        "logo_derecho": formato.logo_derecho,
        "predeterminado": bool(formato.predeterminado),
        "fecha_actualizacion": formato.fecha_actualizacion,
    }


def _propio(formato_id: int, usuario: Usuario, db: Session) -> FormatoImpresion:
    formato = db.query(FormatoImpresion).filter(FormatoImpresion.id == formato_id).first()
    if formato is None or formato.docente_id != usuario.id:
        raise HTTPException(status_code=404, detail="Formato no encontrado.")
    return formato


def _quitar_predeterminado(usuario: Usuario, db: Session, excepto: int | None = None):
    consulta = db.query(FormatoImpresion).filter(FormatoImpresion.docente_id == usuario.id, FormatoImpresion.predeterminado == True)  # noqa: E712
    for otro in consulta:
        if otro.id != excepto:
            otro.predeterminado = False


def _aplicar(formato: FormatoImpresion, datos: Formato):
    formato.nombre = datos.nombre
    formato.datos = datos.datos.model_dump()
    formato.logo_izquierdo = datos.logo_izquierdo
    formato.logo_derecho = datos.logo_derecho
    formato.predeterminado = datos.predeterminado
    formato.fecha_actualizacion = ahora_utc()


def borrar_formatos(db: Session, docente_id: int):
    db.query(FormatoImpresion).filter(FormatoImpresion.docente_id == docente_id).delete(synchronize_session=False)


@router.get("")
def listar(usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    formatos = db.query(FormatoImpresion).filter(FormatoImpresion.docente_id == usuario.id) \
        .order_by(FormatoImpresion.predeterminado.desc(), FormatoImpresion.nombre).all()
    return {"formatos": [formato_a_dict(f) for f in formatos]}


@router.post("")
def crear(datos: Formato, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    if db.query(FormatoImpresion).filter(FormatoImpresion.docente_id == usuario.id).count() >= MAX_FORMATOS:
        raise HTTPException(status_code=400, detail=f"Puedes tener hasta {MAX_FORMATOS} formatos. Borra alguno para crear otro.")
    formato = FormatoImpresion(docente_id=usuario.id, fecha_creacion=ahora_utc())
    _aplicar(formato, datos)
    db.add(formato)
    db.flush()
    if formato.predeterminado:
        _quitar_predeterminado(usuario, db, excepto=formato.id)
    db.commit()
    db.refresh(formato)
    return {"formato": formato_a_dict(formato)}


@router.put("/{formato_id}")
def actualizar(formato_id: int, datos: Formato, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    formato = _propio(formato_id, usuario, db)
    _aplicar(formato, datos)
    if formato.predeterminado:
        _quitar_predeterminado(usuario, db, excepto=formato.id)
    db.commit()
    db.refresh(formato)
    return {"formato": formato_a_dict(formato)}


@router.post("/{formato_id}/duplicar")
def duplicar(formato_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    original = _propio(formato_id, usuario, db)
    if db.query(FormatoImpresion).filter(FormatoImpresion.docente_id == usuario.id).count() >= MAX_FORMATOS:
        raise HTTPException(status_code=400, detail=f"Puedes tener hasta {MAX_FORMATOS} formatos. Borra alguno para crear otro.")
    copia = FormatoImpresion(
        docente_id=usuario.id,
        nombre=f"{original.nombre} (copia)"[:80],
        datos=original.datos,
        logo_izquierdo=original.logo_izquierdo,
        logo_derecho=original.logo_derecho,
        predeterminado=False,
        fecha_creacion=ahora_utc(),
        fecha_actualizacion=ahora_utc(),
    )
    db.add(copia)
    db.commit()
    db.refresh(copia)
    return {"formato": formato_a_dict(copia)}


@router.delete("/{formato_id}")
def eliminar(formato_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    db.delete(_propio(formato_id, usuario, db))
    db.commit()
    return {"mensaje": "Formato eliminado."}
