import os
import re
import sys
import time
import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
import bcrypt
from typing import Literal
from pydantic import BaseModel, Field
from sqlalchemy import func

load_dotenv()

# Los mensajes del servidor llevan acentos: si la consola no es UTF-8 (p. ej. logs redirigidos
# a un archivo en Windows), un print no debe tumbar la petición.
for flujo in (sys.stdout, sys.stderr):
    if hasattr(flujo, "reconfigure"):
        flujo.reconfigure(errors="replace")

# --- IMPORTACIONES DE BASE DE DATOS ---
from sqlalchemy.orm import Session
from database import Cuestionario, Usuario, Intento, InicioIntento, ahora_utc, get_db

# --- CALIFICACIÓN EN EL SERVIDOR ---
from calificacion import calificar, ocultar_respuestas
import identificadores
from identificadores import normalizar_preguntas, respuestas_a_ids, respuestas_para_mostrar

# --- CLASES (grupos) ---
import clases
import reportes
import valoraciones
import resumen_docente
import practicas
import configuracion
import regeneracion
import formatos
import hojas
import versiones
import recuperacion
import respuestas_cortas

# --- AUTENTICACIÓN (JWT) ---
from auth import crear_token, usuario_actual, docente_actual, exigir_mismo_usuario, cuestionario_propio

# --- RAG: recuperación (rag.py) y generación (generacion.py) ---
from rag import construir_contexto, DocumentoSinTexto
from generacion import generar_preguntas, agregar_fuentes, RespuestaInvalida, COMPLEJIDAD_PREDETERMINADA, tipos_validos

Complejidad = Literal["basico", "equilibrado", "avanzado"]

# Modelos para recibir datos desde React
class RegistroUsuario(BaseModel):
    nombre: str
    correo: str
    password: str
    rol: Literal["docente", "estudiante"]

class LoginUsuario(BaseModel):
    correo: str
    password: str

class ActualizarCuestionario(BaseModel):
    preguntas_json: list

class Publicacion(BaseModel):
    # None: solo publica. Una lista: además deja el examen asignado exactamente a esas clases.
    clases: list[int] | None = Field(default=None, max_length=200)

class CambioPassword(BaseModel):
    password_actual: str
    password_nueva: str = Field(min_length=8, max_length=128)  # = MIN_PASSWORD

class ConfirmarPassword(BaseModel):
    password: str

class EntregaIntento(BaseModel):
    # { "<índice de pregunta>": "<id de la opción elegida>" } (se acepta el texto de la opción por compatibilidad)
    respuestas: dict[str, str]
    tiempo_segundos: int | None = Field(default=None, ge=0)
    # Opcional, no cambia la calificación: { "<índice>": true = seguro, false = adivinando }
    seguridad: dict[str, bool] = Field(default_factory=dict)
    # Inicio registrado con POST .../inicio (4.1): el servidor mide el tiempo desde ahí
    inicio_id: int | None = None

class Preferencias(BaseModel):
    complejidad_preferida: Complejidad

class PerfilUpdate(BaseModel):
    nombre: str
    biografia: str | None = None
    institucion: str | None = None
    especialidad: str | None = None
    foto_perfil: str | None = None

app = FastAPI(title="API Backend Tesis E-Learning", version="1.0")

# Orígenes del frontend que pueden llamar a la API, separados por comas en CORS_ORIGINS
# (p. ej. "https://quizai.vercel.app,http://localhost:5173"). Sin la variable, solo desarrollo local.
ORIGENES_PERMITIDOS = [
    origen.strip().rstrip("/")
    for origen in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origen.strip()
]

# La autenticación va en el encabezado Authorization (no en cookies): no hacen falta credenciales.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGENES_PERMITIDOS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(clases.router)
app.include_router(reportes.router)
app.include_router(valoraciones.router)
app.include_router(resumen_docente.router)
app.include_router(practicas.router)
app.include_router(configuracion.router)
app.include_router(regeneracion.router)
app.include_router(formatos.router)
app.include_router(hojas.router)
app.include_router(recuperacion.router)
app.include_router(respuestas_cortas.router)

# 4.6: ids estables en las opciones de los exámenes y en las respuestas de los intentos antiguos.
# Es idempotente; si falla, la app arranca igual (los datos sin ids se siguen calificando por texto).
def migrar_identificadores():
    from database import SessionLocal
    db = SessionLocal()
    try:
        cambios = identificadores.migrar(db)
        cambios["reportes"] = reportes.completar_ids(db)
        if any(cambios.values()):
            print(f"[migracion] Ids de opciones y preguntas: {cambios['cuestionarios']} cuestionarios, "
                  f"{cambios['intentos']} intentos, {cambios['reportes']} reportes")
    except Exception as e:
        db.rollback()
        print(f"AVISO: no se pudieron migrar los ids de opciones: {e}")
    finally:
        db.close()

migrar_identificadores()

CREDITOS_DIARIOS = 5

def regenerar_creditos(usuario: Usuario, db: Session):
    """Evaluación perezosa: si el último uso fue antes de hoy, restablece los créditos del día."""
    hoy = datetime.date.today()
    if usuario.fecha_ultimo_uso is None or usuario.fecha_ultimo_uso < hoy:
        usuario.creditos_disponibles = CREDITOS_DIARIOS
        usuario.regeneraciones_usadas = 0
        usuario.fecha_ultimo_uso = hoy
        db.commit()

def usuario_a_dict(usuario: Usuario) -> dict:
    """Datos del usuario que se envían al frontend (nunca incluye password_hash)."""
    return {
        "id": usuario.id,
        "nombre": usuario.nombre,
        "correo": usuario.correo,
        "rol": usuario.rol,
        "creditos_disponibles": usuario.creditos_disponibles,
        "regeneraciones_restantes": regeneracion.regeneraciones_restantes(usuario),
        "foto_perfil": usuario.foto_perfil,
        "biografia": usuario.biografia,
        "institucion": usuario.institucion,
        "especialidad": usuario.especialidad,
        "complejidad_preferida": usuario.complejidad_preferida or COMPLEJIDAD_PREDETERMINADA,
    }

MAX_TAMANO_PDF = 25 * 1024 * 1024  # 25 MB (el mismo límite que muestra el frontend)

# Es "def" y no "async def": FastAPI lo ejecuta en un hilo aparte, así la lectura del PDF,
# los embeddings y la llamada a Gemini (que tardan segundos) no bloquean al resto de usuarios.
@app.post("/api/generar-cuestionario")
def generar_cuestionario(
    archivo: UploadFile = File(...),
    num_preguntas: int = Form(5, ge=1, le=15),
    nombre_examen: str = Form(...),
    materia: str = Form(...),
    complejidad: Complejidad = Form(COMPLEJIDAD_PREDETERMINADA),
    # 4.2: tipos de pregunta separados por comas: opcion_multiple, verdadero_falso, respuesta_corta
    tipos: str = Form("opcion_multiple"),
    usuario: Usuario = Depends(docente_actual),
    db: Session = Depends(get_db)
):
    # --- 1. CRÉDITOS (regeneración perezosa y saldo) ---
    regenerar_creditos(usuario, db)
    if usuario.creditos_disponibles <= 0:
        raise HTTPException(
            status_code=403,
            detail="Has agotado tus créditos de IA. Por favor, espera a la medianoche para que tu saldo se regenere automáticamente."
        )

    # --- 2. VALIDACIÓN DEL ARCHIVO (en memoria, sin archivos temporales) ---
    contenido = archivo.file.read(MAX_TAMANO_PDF + 1)
    if len(contenido) > MAX_TAMANO_PDF:
        raise HTTPException(status_code=413, detail="El PDF supera el límite de 25 MB.")
    if not (archivo.filename or "").lower().endswith(".pdf") or not contenido.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF válido.")

    # --- 3. RECUPERACIÓN (RAG por cobertura) ---
    try:
        recuperado = construir_contexto(contenido)
    except DocumentoSinTexto as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"ERROR al leer el PDF: {e}")
        raise HTTPException(status_code=400, detail="No se pudo leer el PDF. Verifica que no esté dañado o protegido.")
    print(f"INFO RAG: {recuperado['fragmentos_usados']} de {recuperado['fragmentos_totales']} fragmentos seleccionados por cobertura")

    # --- 4. GENERACIÓN (Gemini) Y VALIDACIÓN ---
    try:
        preguntas = generar_preguntas(recuperado["contexto"], num_preguntas, complejidad, tipos=tipos_validos(tipos.split(",")))
        preguntas = normalizar_preguntas(agregar_fuentes(preguntas, recuperado.get("fragmentos", [])))
        print(f"INFO fuentes: {sum('fuente' in p for p in preguntas)} de {len(preguntas)} preguntas con cita al documento")
    except RespuestaInvalida as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        print(f"ERROR al generar con la IA: {e}")
        raise HTTPException(status_code=502, detail="El servicio de IA no respondió correctamente. Intenta de nuevo en unos minutos.")

    # --- 5. GUARDADO (con reintentos: Neon puede estar "dormido") ---
    max_reintentos = 3
    for intento in range(max_reintentos):
        try:
            nuevo_registro = Cuestionario(
                nombre_examen=nombre_examen,
                nombre_documento=archivo.filename,
                materia=materia,
                preguntas_json=preguntas,
                usuario_id=usuario.id,
                compartido_comunidad=False,
                complejidad=complejidad,
            )
            db.add(nuevo_registro)
            db.commit()
            db.refresh(nuevo_registro)
            break
        except Exception as e:
            db.rollback()
            if intento < max_reintentos - 1:
                print("AVISO: Neon estaba dormido o cortó la conexión. Reintentando en 2 segundos...")
                time.sleep(2)
            else:
                print(f"ERROR crítico en base de datos tras {max_reintentos} intentos: {e}")
                raise HTTPException(status_code=500, detail="Error de conexión con la base de datos.")

    # --- 6. COBRO DEL CRÉDITO (solo si todo salió bien) ---
    usuario.creditos_disponibles -= 1
    usuario.fecha_ultimo_uso = datetime.date.today()
    db.commit()
    return {
        "status": "success",
        "mensaje": "Cuestionario generado correctamente",
        "creditos_disponibles": usuario.creditos_disponibles,
        "preguntas_generadas": len(preguntas),
        "fragmentos_totales": recuperado["fragmentos_totales"],
        "fragmentos_usados": recuperado["fragmentos_usados"],
    }


@app.get("/api/usuarios/{usuario_id}/creditos")
def obtener_creditos(usuario_id: int, usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    """Saldo de créditos del día (ya regenerado si cambió la fecha). Solo los propios."""
    exigir_mismo_usuario(usuario_id, usuario)
    regenerar_creditos(usuario, db)
    return {"creditos_disponibles": usuario.creditos_disponibles}

@app.put("/api/usuarios/{usuario_id}/perfil")
def actualizar_perfil(usuario_id: int, datos: PerfilUpdate, usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    exigir_mismo_usuario(usuario_id, usuario)

    usuario.nombre = datos.nombre
    usuario.biografia = datos.biografia
    usuario.institucion = datos.institucion
    usuario.especialidad = datos.especialidad
    
    # Solo actualizamos la foto si el frontend envió una nueva
    if datos.foto_perfil:
        usuario.foto_perfil = datos.foto_perfil
        
    db.commit()
    db.refresh(usuario)
    
    # Retornamos el usuario actualizado para que React actualice la sesión
    return {
        "status": "success",
        "usuario": usuario_a_dict(usuario)
    }

MIN_PASSWORD = 8
CORREO_VALIDO = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
CREDENCIALES_INVALIDAS = "Correo o contraseña incorrectos."
HASH_FICTICIO = bcrypt.hashpw(b"contrasena-ficticia", bcrypt.gensalt())

def normalizar_correo(correo: str) -> str:
    return correo.strip().lower()

def buscar_por_correo(correo_normalizado: str, db: Session) -> Usuario | None:
    """Sin distinguir mayúsculas (las cuentas antiguas pudieron guardarse con mayúsculas)."""
    return db.query(Usuario).filter(func.lower(Usuario.correo) == correo_normalizado).first()

def verificar_password(password: str, usuario: Usuario) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), usuario.password_hash.encode('utf-8'))

def encriptar_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

@app.put("/api/usuarios/{usuario_id}/password")
def cambiar_password(usuario_id: int, datos: CambioPassword, usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    exigir_mismo_usuario(usuario_id, usuario)
    if not verificar_password(datos.password_actual, usuario):
        raise HTTPException(status_code=400, detail="La contraseña actual no es correcta.")
    if datos.password_nueva == datos.password_actual:
        raise HTTPException(status_code=400, detail="La nueva contraseña debe ser distinta de la actual.")
    usuario.password_hash = encriptar_password(datos.password_nueva)
    db.commit()
    return {"mensaje": "Contraseña actualizada correctamente."}

@app.delete("/api/usuarios/{usuario_id}")
def eliminar_cuenta(usuario_id: int, datos: ConfirmarPassword, usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    """
    Borra la cuenta y sus datos, previa confirmación con la contraseña.
    Docente: sus cuestionarios (con sus intentos y asignaciones) y sus clases.
    Estudiante: sus intentos y sus membresías en clases.
    Las copias que otros docentes clonaron de la comunidad son de ellos y se conservan.
    """
    exigir_mismo_usuario(usuario_id, usuario)
    if not verificar_password(datos.password, usuario):
        raise HTTPException(status_code=400, detail="La contraseña no es correcta.")
    try:
        if usuario.rol == "docente":
            ids = [fila.id for fila in db.query(Cuestionario.id).filter(Cuestionario.usuario_id == usuario.id)]
            if ids:
                db.query(Intento).filter(Intento.cuestionario_id.in_(ids)).delete(synchronize_session=False)
                reportes.borrar_reportes(db, cuestionario_ids=ids)
                practicas.borrar_practicas(db, cuestionario_ids=ids)
                configuracion.borrar_inicios(db, cuestionario_ids=ids)
                clases.borrar_asignaciones_de_cuestionarios(ids, db)
                db.query(Cuestionario).filter(Cuestionario.id.in_(ids)).delete(synchronize_session=False)
            clases.borrar_clases_de_docente(usuario.id, db)
            formatos.borrar_formatos(db, usuario.id)
        else:
            db.query(Intento).filter(Intento.estudiante_id == usuario.id).delete(synchronize_session=False)
            reportes.borrar_reportes(db, estudiante_id=usuario.id)
            practicas.borrar_practicas(db, estudiante_id=usuario.id)
            configuracion.borrar_inicios(db, estudiante_id=usuario.id)
            clases.borrar_membresias_de_estudiante(usuario.id, db)
        recuperacion.borrar_restablecimientos(db, usuario.id)
        db.delete(usuario)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"ERROR al eliminar la cuenta {usuario_id}: {e}")
        raise HTTPException(status_code=500, detail="No se pudo eliminar la cuenta. Intenta de nuevo.")
    return {"mensaje": "Cuenta eliminada."}

@app.put("/api/usuarios/{usuario_id}/preferencias")
def guardar_preferencias(usuario_id: int, datos: Preferencias, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    """Docente: complejidad predeterminada del generador IA."""
    exigir_mismo_usuario(usuario_id, usuario)
    usuario.complejidad_preferida = datos.complejidad_preferida
    db.commit()
    return {"complejidad_preferida": usuario.complejidad_preferida}

@app.get("/api/cuestionarios")
def obtener_historial(usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    """Docente: sus cuestionarios. Estudiante: los publicados en sus clases. (El rol sale del token.)"""
    try:
        if usuario.rol == 'estudiante':
            # El estudiante no recibe la clave ni la justificación: la corrección llega al entregar
            configuracion.entregar_vencidos(db, estudiante_id=usuario.id)
            ahora = ahora_utc()
            historial = [
                {**cuestionario_para_estudiante(c), "clases": nombres_clases,
                 # Intentos, cierre y si puede iniciar (4.1)
                 "mi_estado": configuracion.estado_estudiante(c, usuario.id, db, ahora)}
                for c, nombres_clases in clases.cuestionarios_visibles(usuario.id, db)
            ]
        else:
            historial = db.query(Cuestionario).filter(Cuestionario.usuario_id == usuario.id).order_by(Cuestionario.id.desc()).all()
        return {"data": historial}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al obtener el historial.")

@app.get("/api/cuestionarios/{cuestionario_id}/clases")
def clases_del_cuestionario(cuestionario_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    """Ids de las clases a las que está asignado el cuestionario (para el diálogo de publicar)."""
    cuestionario = cuestionario_propio(cuestionario_id, usuario, db)
    return {"clases": clases.ids_clases_de_cuestionario(cuestionario.id, db)}

@app.put("/api/cuestionarios/{cuestionario_id}/publicar")
def publicar_cuestionario(cuestionario_id: int, datos: Publicacion | None = None, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    """Publica el cuestionario y, si se envían `clases`, lo asigna a ellas en la misma operación."""
    cuestionario = cuestionario_propio(cuestionario_id, usuario, db)
    if datos is not None and datos.clases is not None:
        clases.sincronizar_asignaciones(cuestionario, datos.clases, usuario, db)
    cuestionario.publicado = True
    db.commit()
    return {"mensaje": "Cuestionario publicado exitosamente", "clases": clases.ids_clases_de_cuestionario(cuestionario.id, db)}

@app.put("/api/cuestionarios/{cuestionario_id}/despublicar")
def despublicar_cuestionario(cuestionario_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    cuestionario = cuestionario_propio(cuestionario_id, usuario, db)
    cuestionario.publicado = False
    db.commit()
    return {"mensaje": "Cuestionario ocultado exitosamente"}

@app.delete("/api/cuestionarios/{cuestionario_id}")
def eliminar_cuestionario(cuestionario_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    cuestionario = cuestionario_propio(cuestionario_id, usuario, db)
    db.query(Intento).filter(Intento.cuestionario_id == cuestionario.id).delete()
    reportes.borrar_reportes(db, cuestionario_ids=[cuestionario.id])
    practicas.borrar_practicas(db, cuestionario_ids=[cuestionario.id])
    configuracion.borrar_inicios(db, cuestionario_ids=[cuestionario.id])
    clases.borrar_asignaciones_de_cuestionarios([cuestionario.id], db)
    db.delete(cuestionario)
    db.commit()
    return {"mensaje": "Cuestionario eliminado exitosamente"}

@app.put("/api/cuestionarios/{id}")
def actualizar_cuestionario(id: int, datos: ActualizarCuestionario, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    cuestionario = cuestionario_propio(id, usuario, db)
    # Conserva los ids de las opciones y recalcula id_correcta (4.6)
    cuestionario.preguntas_json = normalizar_preguntas(datos.preguntas_json)
    # Los reportes siguen a su pregunta (por id); los de preguntas borradas o regeneradas se eliminan
    reportes.sincronizar_reportes(db, cuestionario)
    db.commit()
    # Se devuelven las preguntas con sus ids: el editor las adopta para que un segundo guardado no los cambie
    return {"mensaje": "Cuestionario actualizado correctamente", "preguntas_json": cuestionario.preguntas_json}


# ==========================================================
# INTENTOS DE LOS ESTUDIANTES (calificados en el servidor)
# ==========================================================

def cuestionario_para_estudiante(cuestionario: Cuestionario) -> dict:
    return {
        "id": cuestionario.id,
        "nombre_examen": cuestionario.nombre_examen,
        "nombre_documento": cuestionario.nombre_documento,
        "materia": cuestionario.materia,
        "publicado": cuestionario.publicado,
        "fecha_creacion": cuestionario.fecha_creacion,
        "preguntas_json": ocultar_respuestas(cuestionario.preguntas_json or []),
        "configuracion": configuracion.configuracion_a_dict(cuestionario),
    }

def preguntas_con_ids(cuestionario: Cuestionario, db: Session) -> list:
    """Las preguntas del examen; si alguna no tiene ids (datos antiguos), se los asigna y guarda."""
    preguntas = cuestionario.preguntas_json or []
    if not identificadores.tiene_ids(preguntas):
        cuestionario.preguntas_json = preguntas = normalizar_preguntas(preguntas)
        db.commit()
    return preguntas

def intento_a_dict(intento: Intento, preguntas: list, con_correccion: bool = True) -> dict:
    """
    Las respuestas se guardan como ids de opción; aquí se traducen al índice y al texto ACTUALES de
    cada pregunta (el frontend compara textos). Así, editar una opción o borrar una pregunta después
    de las entregas no descuadra los intentos anteriores.
    con_correccion=False (respuestas ocultas, 4.1): de las respuestas cortas solo va lo que escribió.
    """
    respuestas, seguridad = respuestas_para_mostrar(preguntas, intento.respuestas, intento.seguridad, intento.respuestas_cortas)
    cortas = identificadores.cortas_actuales(preguntas, intento.respuestas_cortas)
    return {
        "id": intento.id,
        "cuestionario_id": intento.cuestionario_id,
        "estudiante_id": intento.estudiante_id,
        "respuestas": respuestas,
        "puntos_obtenidos": intento.puntos_obtenidos,
        "puntos_totales": intento.puntos_totales,
        "calificacion": intento.calificacion,
        "tiempo_segundos": intento.tiempo_segundos,
        "fecha": intento.fecha,
        "seguridad": seguridad,
        "fuera_de_tiempo": bool(intento.fuera_de_tiempo),
        "entrega_automatica": bool(intento.entrega_automatica),
        "en_papel": bool(intento.en_papel),
        # 4.2: { "<índice>": {texto, puntaje, comentario, estado} } de las respuestas cortas
        "respuestas_cortas": cortas if con_correccion else respuestas_cortas.sin_correccion(cortas),
        # 2.6: orden en que vio preguntas y opciones (None = el original), ajustado a las preguntas actuales
        "orden": versiones.normalizar_orden(intento.orden, preguntas) if intento.orden else None,
    }

@app.post("/api/cuestionarios/{cuestionario_id}/intentos")
def entregar_intento(cuestionario_id: int, datos: EntregaIntento, usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    """El estudiante entrega sus respuestas; el servidor califica, guarda el intento y devuelve la corrección."""
    if usuario.rol != "estudiante":
        raise HTTPException(status_code=403, detail="Solo los estudiantes pueden entregar evaluaciones.")
    cuestionario = db.query(Cuestionario).filter(Cuestionario.id == cuestionario_id).first()
    if cuestionario is None or not cuestionario.publicado or not clases.estudiante_puede_ver(usuario.id, cuestionario.id, db):
        raise HTTPException(status_code=404, detail="Esta evaluación ya no está disponible.")

    # Inicio registrado (4.1): obligatorio con tiempo límite o intentos máximos
    ahora = ahora_utc()
    inicio = None
    if datos.inicio_id is not None:
        inicio = db.query(InicioIntento).filter(InicioIntento.id == datos.inicio_id).first()
        if inicio is None or inicio.estudiante_id != usuario.id or inicio.cuestionario_id != cuestionario.id:
            raise HTTPException(status_code=400, detail="No se encontró el inicio de esta evaluación. Vuelve a iniciarla.")
        if inicio.entregado:
            raise HTTPException(status_code=409, detail="Este intento ya se entregó.")
    elif configuracion.requiere_inicio(cuestionario):
        raise HTTPException(status_code=400, detail="Esta evaluación tiene tiempo o intentos limitados: vuelve a iniciarla.")
    tiempo_servidor, fuera_de_tiempo = configuracion.evaluar_entrega(cuestionario, inicio, ahora)

    preguntas = preguntas_con_ids(cuestionario, db)
    # Solo se guardan respuestas a preguntas que existen, como ids de opción
    respuestas = respuestas_a_ids(preguntas, datos.respuestas)
    # La seguridad solo cuenta en preguntas respondidas
    seguridad = {k: v for k, v in datos.seguridad.items() if k in respuestas}
    # 4.2: las respuestas cortas se califican con la IA (una sola llamada por entrega)
    cortas = respuestas_cortas.corregir(preguntas, respuestas)
    resultado = calificar(preguntas, respuestas, cortas)

    intento = Intento(
        cuestionario_id=cuestionario.id,
        estudiante_id=usuario.id,
        respuestas=respuestas,
        puntos_obtenidos=resultado["puntos_obtenidos"],
        puntos_totales=resultado["puntos_totales"],
        calificacion=resultado["calificacion"],
        # Medido en el servidor cuando hay inicio; si no, el que informa el navegador
        tiempo_segundos=tiempo_servidor if tiempo_servidor is not None else datos.tiempo_segundos,
        seguridad=seguridad,
        respuestas_cortas=cortas,
        inicio_id=inicio.id if inicio else None,
        orden=inicio.orden if inicio else None,  # 2.6: la revisión lo muestra en el orden en que lo vio
        fuera_de_tiempo=fuera_de_tiempo,
    )
    if inicio is not None:
        inicio.entregado = True
    try:
        db.add(intento)
        db.commit()
        db.refresh(intento)
    except Exception as e:
        db.rollback()
        print(f"ERROR al guardar el intento: {e}")
        raise HTTPException(status_code=500, detail="No se pudo guardar tu evaluación. Intenta entregarla de nuevo.")

    visibles = configuracion.respuestas_visibles(cuestionario, ahora)
    return {
        "intento": intento_a_dict(intento, preguntas, con_correccion=visibles),
        # Si el docente no muestra las respuestas (todavía), solo va la calificación
        "respuestas_visibles": visibles,
        "respuestas_correctas": resultado["respuestas_correctas"] if visibles else None,
        "justificaciones": resultado["justificaciones"] if visibles else None,
        "fuentes": resultado["fuentes"] if visibles else None,
        "configuracion": configuracion.configuracion_a_dict(cuestionario),
    }

# Va antes de /api/intentos/{intento_id} para que "mios" no se interprete como un id.
@app.get("/api/intentos/mios")
def mis_intentos(usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    """Estudiante: sus intentos (más reciente primero), con el nombre y la materia del examen."""
    if usuario.rol != "estudiante":
        raise HTTPException(status_code=403, detail="Esta sección es solo para estudiantes.")
    configuracion.entregar_vencidos(db, estudiante_id=usuario.id)
    filas = db.query(Intento, Cuestionario) \
        .join(Cuestionario, Cuestionario.id == Intento.cuestionario_id) \
        .filter(Intento.estudiante_id == usuario.id) \
        .order_by(Intento.fecha.desc(), Intento.id.desc()).all()
    return {"data": [
        {**intento_a_dict(intento, cuestionario.preguntas_json or [],
                          con_correccion=configuracion.respuestas_visibles(cuestionario, ahora_utc())),
         "nombre_examen": cuestionario.nombre_examen or cuestionario.nombre_documento, "materia": cuestionario.materia}
        for intento, cuestionario in filas
    ]}

@app.get("/api/intentos/{intento_id}")
def revisar_intento(intento_id: int, usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    """
    Revisión de un intento ya entregado: respuestas elegidas, clave y retroalimentación.
    Lo pueden ver el estudiante que lo hizo y el docente dueño del examen.
    """
    intento = db.query(Intento).filter(Intento.id == intento_id).first()
    cuestionario = intento and db.query(Cuestionario).filter(Cuestionario.id == intento.cuestionario_id).first()
    if intento is None or cuestionario is None:
        raise HTTPException(status_code=404, detail="Intento no encontrado.")
    if usuario.id not in (intento.estudiante_id, cuestionario.usuario_id):
        raise HTTPException(status_code=403, detail="No puedes ver este intento.")
    es_docente = usuario.id == cuestionario.usuario_id
    visibles = es_docente or configuracion.respuestas_visibles(cuestionario, ahora_utc())
    if es_docente:
        preguntas_mostradas = cuestionario.preguntas_json or []
    elif visibles:
        preguntas_mostradas = valoraciones.sin_valoraciones(cuestionario.preguntas_json)
    else:
        preguntas_mostradas = ocultar_respuestas(cuestionario.preguntas_json or [])
    return {
        "intento": intento_a_dict(intento, cuestionario.preguntas_json or [], con_correccion=visibles),
        "examen": {
            "id": cuestionario.id,
            "nombre_examen": cuestionario.nombre_examen or cuestionario.nombre_documento,
            "materia": cuestionario.materia,
            # La valoración del docente (calidad del generador) no es para el estudiante; la clave,
            # solo si el docente muestra las respuestas (4.1)
            "preguntas_json": preguntas_mostradas,
            "configuracion": configuracion.configuracion_a_dict(cuestionario),
        },
        "respuestas_visibles": visibles,
        # Preguntas de este examen que el estudiante del intento ya reportó
        "preguntas_reportadas": reportes.preguntas_reportadas(intento.estudiante_id, cuestionario.id, db),
    }

@app.get("/api/cuestionarios/{cuestionario_id}/intentos")
def listar_intentos(cuestionario_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    """
    Docente: todos los intentos de uno de sus cuestionarios, con el nombre del estudiante.
    `clases` son las clases a las que está asignado; cada intento trae `clase_ids`, las de esas
    clases en las que el estudiante está inscrito hoy (para filtrar por clase).
    """
    cuestionario = cuestionario_propio(cuestionario_id, usuario, db)
    configuracion.entregar_vencidos(db, cuestionario_ids=[cuestionario.id])
    filas = db.query(Intento, Usuario.nombre, Usuario.correo) \
        .join(Usuario, Intento.estudiante_id == Usuario.id) \
        .filter(Intento.cuestionario_id == cuestionario.id) \
        .order_by(Intento.fecha.desc(), Intento.id.desc()).all()
    clases_asignadas, clases_por_estudiante = clases.clases_de_cuestionario(cuestionario.id, db)
    return {
        "data": [
            {
                **intento_a_dict(intento, cuestionario.preguntas_json or []),
                "nombre_estudiante": nombre,
                "correo_estudiante": correo,
                "clase_ids": clases_por_estudiante.get(intento.estudiante_id, []),
            }
            for intento, nombre, correo in filas
        ],
        "clases": clases_asignadas,
        "reportes": reportes.reportes_de_cuestionario(cuestionario.id, db),
        # Prácticas adaptativas terminadas por estudiante: el docente solo ve el conteo
        "practicas": practicas.practicas_terminadas_por_estudiante(cuestionario.id, db),
    }


# ==========================================================
# NUEVOS ENDPOINTS: MERCADO COMUNITARIO DE DOCENTES
# ==========================================================

@app.get("/api/comunidad/cuestionarios")
def obtener_cuestionarios_comunidad(usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    """Obtiene todos los exámenes compartidos públicamente haciendo un JOIN para traer el nombre del autor"""
    try:
        registro_compartidos = db.query(
            Cuestionario.id,
            Cuestionario.nombre_examen,
            Cuestionario.materia,
            Cuestionario.preguntas_json,
            Cuestionario.nombre_documento,
            Usuario.nombre.label("nombre_autor")
        ).join(Usuario, Cuestionario.usuario_id == Usuario.id).filter(Cuestionario.compartido_comunidad == True).order_by(Cuestionario.id.desc()).all()
        
        # Mapeamos el resultado a un formato JSON limpio que entienda React
        comunidad_data = []
        for item in registro_compartidos:
            comunidad_data.append({
                "id": item.id,
                "nombre_examen": item.nombre_examen,
                "materia": item.materia,
                "nombre_documento": item.nombre_documento,
                "autor": item.nombre_autor,
                "num_preguntas": len(item.preguntas_json) if item.preguntas_json else 0
            })
            
        return {"data": comunidad_data}
    except Exception as e:
        print(f"ERROR en Comunidad: {e}")
        raise HTTPException(status_code=500, detail="Error al cargar la comunidad.")

@app.put("/api/cuestionarios/{cuestionario_id}/compartir")
def compartir_en_comunidad(cuestionario_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    """Cambia el interruptor para hacer el examen público en el mercado comunitario"""
    cuestionario = cuestionario_propio(cuestionario_id, usuario, db)
    cuestionario.compartido_comunidad = True
    db.commit()
    return {"status": "success", "mensaje": "Examen compartido en la comunidad con éxito"}

@app.post("/api/cuestionarios/{cuestionario_id}/clonar")
def clonar_cuestionario(cuestionario_id: int, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    """Toma un examen de la comunidad (o uno propio), crea un clon añadiendo '(Copia)' y se lo asigna al docente del token"""
    cuestionario_original = db.query(Cuestionario).filter(Cuestionario.id == cuestionario_id).first()
    if not cuestionario_original:
        raise HTTPException(status_code=404, detail="Cuestionario base no encontrado")
    # Solo se pueden clonar exámenes compartidos en la comunidad (o los propios), no los privados de otros
    if not cuestionario_original.compartido_comunidad and cuestionario_original.usuario_id != usuario.id:
        raise HTTPException(status_code=403, detail="Este cuestionario no está compartido en la comunidad.")

    try:
        nuevo_clon = Cuestionario(
            nombre_examen=f"{cuestionario_original.nombre_examen} (Copia)",
            nombre_documento=cuestionario_original.nombre_documento,
            materia=cuestionario_original.materia,
            complejidad=cuestionario_original.complejidad,
            preguntas_json=valoraciones.sin_valoraciones(cuestionario_original.preguntas_json),
            usuario_id=usuario.id,       # <--- Se asigna al docente que lo clonó (del token)
            publicado=False,             # Nace como borrador privado
            compartido_comunidad=False   # No se vuelve a auto-compartir
        )
        db.add(nuevo_clon)
        db.commit()
        db.refresh(nuevo_clon)
        return {"status": "success", "mensaje": "Examen clonado exitosamente"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"No se pudo clonar el examen: {str(e)}")


# ==========================================
# RUTAS DE AUTENTICACIÓN (LOGIN Y REGISTRO)
# ==========================================

@app.post("/api/auth/register")
def registrar_usuario(datos: RegistroUsuario, db: Session = Depends(get_db)):
    nombre = datos.nombre.strip()
    correo = normalizar_correo(datos.correo)
    if not nombre:
        raise HTTPException(status_code=400, detail="Escribe tu nombre.")
    if not CORREO_VALIDO.fullmatch(correo):
        raise HTTPException(status_code=400, detail="Escribe un correo electrónico válido.")
    if len(datos.password) < MIN_PASSWORD:
        raise HTTPException(status_code=400, detail=f"La contraseña debe tener al menos {MIN_PASSWORD} caracteres.")
    if buscar_por_correo(correo, db):
        raise HTTPException(status_code=400, detail="El correo ya está registrado")

    nuevo_usuario = Usuario(
        nombre=nombre,
        correo=correo,
        password_hash=encriptar_password(datos.password),
        rol=datos.rol
    )
    db.add(nuevo_usuario)
    db.commit()
    db.refresh(nuevo_usuario)
    
    return {"mensaje": "Usuario registrado exitosamente", "rol": nuevo_usuario.rol}

@app.post("/api/auth/login")
def iniciar_sesion(datos: LoginUsuario, db: Session = Depends(get_db)):
    usuario = buscar_por_correo(normalizar_correo(datos.correo), db)
    # Mismo mensaje (y un bcrypt de igual costo) exista o no el correo: así la respuesta
    # no revela qué correos están registrados.
    if usuario is None:
        bcrypt.checkpw(datos.password.encode('utf-8'), HASH_FICTICIO)
        raise HTTPException(status_code=401, detail=CREDENCIALES_INVALIDAS)
    if not verificar_password(datos.password, usuario):
        raise HTTPException(status_code=401, detail=CREDENCIALES_INVALIDAS)

    regenerar_creditos(usuario, db)
    return {
        "mensaje": "Login exitoso",
        "token": crear_token(usuario),
        "usuario": usuario_a_dict(usuario)
    }