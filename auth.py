"""Autenticación con JWT.

- El login entrega un token firmado (HS256) que identifica al usuario.
- El frontend lo envía en cada petición: `Authorization: Bearer <token>`.
- Los endpoints obtienen al usuario con `Depends(usuario_actual)` o `Depends(docente_actual)`;
  nunca confían en un `usuario_id` enviado por el cliente.
"""
import datetime
import os

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from database import Cuestionario, Usuario, get_db

load_dotenv()
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET or len(JWT_SECRET) < 32:
    raise ValueError(
        "⚠️ Falta JWT_SECRET en el archivo .env (mínimo 32 caracteres). "
        "Genera uno con: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
    )

ALGORITMO = "HS256"
DURACION_TOKEN = datetime.timedelta(days=7)

esquema_bearer = HTTPBearer(auto_error=False)


def crear_token(usuario: Usuario) -> str:
    ahora = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "sub": str(usuario.id),
        "rol": usuario.rol,
        "iat": ahora,
        "exp": ahora + DURACION_TOKEN,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITMO)


def usuario_actual(
    credenciales: HTTPAuthorizationCredentials | None = Depends(esquema_bearer),
    db: Session = Depends(get_db),
) -> Usuario:
    """Usuario dueño del token. 401 si falta, está vencido, alterado o el usuario ya no existe."""
    no_autenticado = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Tu sesión expiró o no es válida. Inicia sesión de nuevo.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credenciales is None:
        raise no_autenticado
    try:
        payload = jwt.decode(credenciales.credentials, JWT_SECRET, algorithms=[ALGORITMO])
        usuario_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise no_autenticado

    usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()
    if usuario is None:
        raise no_autenticado
    return usuario


def docente_actual(usuario: Usuario = Depends(usuario_actual)) -> Usuario:
    """Como usuario_actual, pero solo para docentes (403 para los demás)."""
    if usuario.rol != "docente":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Esta acción es solo para docentes.")
    return usuario


def exigir_mismo_usuario(usuario_id: int, usuario: Usuario):
    """403 si la URL pide datos de otro usuario."""
    if usuario_id != usuario.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No puedes acceder a los datos de otro usuario.")


def cuestionario_propio(cuestionario_id: int, usuario: Usuario, db: Session) -> Cuestionario:
    """El cuestionario si pertenece al usuario; 404 si no existe, 403 si es de otro docente."""
    cuestionario = db.query(Cuestionario).filter(Cuestionario.id == cuestionario_id).first()
    if cuestionario is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cuestionario no encontrado")
    if cuestionario.usuario_id != usuario.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Este cuestionario pertenece a otro docente.")
    return cuestionario
