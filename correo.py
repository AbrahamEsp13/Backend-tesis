"""Envío de correos por SMTP (biblioteca estándar, sin dependencias).

Sirve con cualquier servidor SMTP: Gmail u Outlook (con contraseña de aplicación), o servicios como
Resend, SendGrid o Brevo, que también ofrecen SMTP. Variables del .env:

    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587                 # 587 = STARTTLS; 465 = SSL
    SMTP_USUARIO=cuenta@gmail.com
    SMTP_PASSWORD=contraseña-de-aplicación
    CORREO_REMITENTE="QuizAI <cuenta@gmail.com>"   # opcional (por defecto SMTP_USUARIO)

Sin SMTP_HOST (desarrollo), el correo no se envía: se escribe en la consola del servidor.
"""
import os
import smtplib
import ssl
from email.message import EmailMessage

from dotenv import load_dotenv

load_dotenv()


def configurado() -> bool:
    return bool(os.getenv("SMTP_HOST"))


def enviar(destinatario: str, asunto: str, texto: str, html: str | None = None) -> bool:
    """Envía el correo. Devuelve False (y lo registra) si falla; nunca lanza la excepción."""
    if not configurado():
        print(f"[correo] SMTP sin configurar. Correo para {destinatario} -- {asunto}\n{texto}")
        return False
    mensaje = EmailMessage()
    mensaje["Subject"] = asunto
    mensaje["From"] = os.getenv("CORREO_REMITENTE") or os.getenv("SMTP_USUARIO")
    mensaje["To"] = destinatario
    mensaje.set_content(texto)
    if html:
        mensaje.add_alternative(html, subtype="html")

    host, puerto = os.getenv("SMTP_HOST"), int(os.getenv("SMTP_PORT", "587"))
    usuario, password = os.getenv("SMTP_USUARIO"), os.getenv("SMTP_PASSWORD")
    contexto = ssl.create_default_context()
    try:
        if puerto == 465:
            with smtplib.SMTP_SSL(host, puerto, context=contexto, timeout=20) as servidor:
                if usuario:
                    servidor.login(usuario, password)
                servidor.send_message(mensaje)
        else:
            with smtplib.SMTP(host, puerto, timeout=20) as servidor:
                servidor.starttls(context=contexto)
                if usuario:
                    servidor.login(usuario, password)
                servidor.send_message(mensaje)
        return True
    except Exception as e:
        print(f"ERROR al enviar el correo ({asunto}): {e}")
        return False
