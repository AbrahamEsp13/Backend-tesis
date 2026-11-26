"""4.7 Recuperar la contraseña por correo.

1. POST /api/auth/recuperar {correo}: si la cuenta existe, se envía un enlace de un solo uso que vence
   en VIGENCIA. La respuesta es la misma exista o no la cuenta (no revela qué correos están registrados)
   y el correo se envía en segundo plano (tampoco lo revela el tiempo de respuesta).
2. POST /api/auth/restablecer/verificar {token}: el frontend comprueba el enlace antes de pedir la contraseña.
3. POST /api/auth/restablecer {token, password_nueva}: cambia la contraseña, invalida el enlace y cierra
   las sesiones abiertas (sube `version_sesion`, que va en el token; ver auth.usuario_actual).

En la base solo se guarda el hash SHA-256 del enlace: quien lea la tabla no puede usarlo.
Límite: SOLICITUDES_POR_HORA por cuenta. Pedir un enlace nuevo invalida los anteriores.
"""
import datetime
import hashlib
import os
import secrets

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import correo
from database import RestablecimientoPassword, Usuario, ahora_utc, get_db

router = APIRouter(prefix="/api/auth", tags=["recuperacion"])

VIGENCIA = datetime.timedelta(minutes=30)
SOLICITUDES_POR_HORA = 3
MENSAJE_ENVIADO = ("Si hay una cuenta con ese correo, te enviamos un enlace para restablecer tu contraseña. "
                   "Revisa también la carpeta de correo no deseado.")
ENLACE_INVALIDO = "El enlace no es válido o ya venció. Pide uno nuevo desde \"¿Olvidaste tu contraseña?\"."


class SolicitudRecuperacion(BaseModel):
    correo: str = Field(max_length=254)


class TokenRecuperacion(BaseModel):
    token: str = Field(min_length=20, max_length=200)


class Restablecimiento(TokenRecuperacion):
    password_nueva: str = Field(min_length=8, max_length=128)  # = main.MIN_PASSWORD


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def url_frontend() -> str:
    """Dónde está la app (para armar el enlace). FRONTEND_URL en el .env; si no, el primer origen de CORS."""
    url = os.getenv("FRONTEND_URL") or os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")[0]
    return url.strip().rstrip("/")


def _correo_de_recuperacion(nombre: str, enlace: str) -> tuple[str, str]:
    minutos = int(VIGENCIA.total_seconds() // 60)
    texto = (
        f"Hola, {nombre}:\n\n"
        f"Recibimos una solicitud para restablecer la contraseña de tu cuenta de QuizAI.\n"
        f"Abre este enlace para elegir una nueva (vence en {minutos} minutos y solo funciona una vez):\n\n"
        f"{enlace}\n\n"
        f"Si no fuiste tú, ignora este correo: tu contraseña no cambiará.\n"
    )
    html = f"""<div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;color:#1f2937">
  <h2 style="color:#2563eb">QuizAI</h2>
  <p>Hola, {nombre}:</p>
  <p>Recibimos una solicitud para restablecer la contraseña de tu cuenta.</p>
  <p style="margin:28px 0"><a href="{enlace}" style="background:#2563eb;color:#fff;padding:12px 22px;border-radius:10px;text-decoration:none;font-weight:bold">Elegir una nueva contraseña</a></p>
  <p style="font-size:13px;color:#6b7280">El enlace vence en {minutos} minutos y solo funciona una vez.
  Si no fuiste tú, ignora este correo: tu contraseña no cambiará.</p>
</div>"""
    return texto, html


def _vigente(token: str, db: Session) -> RestablecimientoPassword | None:
    registro = db.query(RestablecimientoPassword).filter(RestablecimientoPassword.token_hash == hash_token(token)).first()
    if registro is None or registro.usado or ahora_utc() > registro.expira:
        return None
    return registro


@router.post("/recuperar")
def recuperar(datos: SolicitudRecuperacion, tareas: BackgroundTasks, db: Session = Depends(get_db)):
    from main import buscar_por_correo, normalizar_correo  # aquí: main importa este módulo
    usuario = buscar_por_correo(normalizar_correo(datos.correo), db)
    if usuario is None:
        return {"mensaje": MENSAJE_ENVIADO}

    ahora = ahora_utc()
    recientes = db.query(RestablecimientoPassword).filter(
        RestablecimientoPassword.usuario_id == usuario.id,
        RestablecimientoPassword.fecha >= ahora - datetime.timedelta(hours=1),
    ).count()
    if recientes >= SOLICITUDES_POR_HORA:
        print(f"AVISO recuperación: límite de solicitudes alcanzado para el usuario {usuario.id}")
        return {"mensaje": MENSAJE_ENVIADO}  # misma respuesta: no se revela nada

    # Un enlace nuevo invalida los anteriores
    db.query(RestablecimientoPassword).filter(
        RestablecimientoPassword.usuario_id == usuario.id, RestablecimientoPassword.usado == False  # noqa: E712
    ).update({"usado": True})
    token = secrets.token_urlsafe(32)
    db.add(RestablecimientoPassword(usuario_id=usuario.id, token_hash=hash_token(token), expira=ahora + VIGENCIA, fecha=ahora))
    db.commit()

    texto, html = _correo_de_recuperacion(usuario.nombre, f"{url_frontend()}/restablecer?token={token}")
    tareas.add_task(correo.enviar, usuario.correo, "Restablece tu contraseña de QuizAI", texto, html)
    return {"mensaje": MENSAJE_ENVIADO}


@router.post("/restablecer/verificar")
def verificar(datos: TokenRecuperacion, db: Session = Depends(get_db)):
    registro = _vigente(datos.token, db)
    if registro is None:
        raise HTTPException(status_code=400, detail=ENLACE_INVALIDO)
    usuario = db.query(Usuario).filter(Usuario.id == registro.usuario_id).first()
    usuario_correo, dominio = (usuario.correo.split("@", 1) + [""])[:2]
    return {"correo": f"{usuario_correo[:2]}{'*' * max(1, len(usuario_correo) - 2)}@{dominio}"}


@router.post("/restablecer")
def restablecer(datos: Restablecimiento, db: Session = Depends(get_db)):
    from main import encriptar_password
    registro = _vigente(datos.token, db)
    if registro is None:
        raise HTTPException(status_code=400, detail=ENLACE_INVALIDO)
    usuario = db.query(Usuario).filter(Usuario.id == registro.usuario_id).first()
    if usuario is None:
        raise HTTPException(status_code=400, detail=ENLACE_INVALIDO)
    usuario.password_hash = encriptar_password(datos.password_nueva)
    # Cierra las sesiones abiertas: los tokens con la versión anterior dejan de valer
    usuario.version_sesion = (usuario.version_sesion or 0) + 1
    registro.usado = True
    db.commit()
    return {"mensaje": "Listo: tu contraseña cambió. Ya puedes iniciar sesión con la nueva."}


def borrar_restablecimientos(db: Session, usuario_id: int):
    db.query(RestablecimientoPassword).filter(RestablecimientoPassword.usuario_id == usuario_id).delete(synchronize_session=False)
