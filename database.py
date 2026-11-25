import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text, Column, Integer, String, JSON, DateTime, Boolean, Date, Float, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

# 1. Cargar la URL de tu archivo .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("No se encontró DATABASE_URL en el archivo .env")

# SQLAlchemy 2.1+ usa 'psycopg' (v3) por defecto para 'postgresql://'.
# Forzamos 'psycopg2', que es el driver instalado en requirements.txt.
for prefijo in ("postgresql://", "postgres://"):
    if DATABASE_URL.startswith(prefijo):
        DATABASE_URL = "postgresql+psycopg2://" + DATABASE_URL[len(prefijo):]
        break

# 2. Configurar el motor de conexión
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def ahora_utc() -> datetime.datetime:
    """Fecha y hora UTC sin zona (como guardaba datetime.utcnow, que está obsoleto)."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

def get_db():
    """Dependencia de FastAPI: una sesión de BD por petición."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 3. Definir la Tabla Entidad-Relación
class Cuestionario(Base):
    __tablename__ = "cuestionarios_v3" 

    id = Column(Integer, primary_key=True, index=True)
    nombre_examen = Column(String) 
    nombre_documento = Column(String, index=True)
    materia = Column(String)                               
    compartido_comunidad = Column(Boolean, default=False)  
    preguntas_json = Column(JSON, nullable=False) 
    publicado = Column(Boolean, default=False) 
    fecha_creacion = Column(DateTime, default=ahora_utc)
    usuario_id = Column(Integer, index=True)
    complejidad = Column(String, nullable=True)  # 'basico' | 'equilibrado' | 'avanzado' (con la que se generó)

class Usuario(Base):
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, index=True)
    correo = Column(String, unique=True, index=True) 
    password_hash = Column(String) 
    rol = Column(String) # 'docente' o 'estudiante'
    fecha_registro = Column(DateTime, default=ahora_utc)
    creditos_disponibles = Column(Integer, default=5)
    fecha_ultimo_uso = Column(Date, default=datetime.date.today)

    foto_perfil = Column(String, nullable=True)
    biografia = Column(String, nullable=True)
    institucion = Column(String, nullable=True)
    especialidad = Column(String, nullable=True)
    complejidad_preferida = Column(String, nullable=True)  # preferencia del generador IA (docentes)

class Intento(Base):
    """Una entrega de un estudiante. Cada reintento es un registro nuevo."""
    __tablename__ = "intentos"

    id = Column(Integer, primary_key=True, index=True)
    cuestionario_id = Column(Integer, index=True, nullable=False)
    estudiante_id = Column(Integer, index=True, nullable=False)
    respuestas = Column(JSON, nullable=False)          # { "<índice>": "<opción elegida>" }
    puntos_obtenidos = Column(Float, nullable=False)
    puntos_totales = Column(Float, nullable=False)
    calificacion = Column(Float, nullable=False)       # sobre 10
    tiempo_segundos = Column(Integer, nullable=True)
    fecha = Column(DateTime, default=ahora_utc)
    # Seguridad que marcó el estudiante (opcional): { "<índice>": true = seguro, false = adivinando }
    seguridad = Column(JSON, nullable=True)

class Clase(Base):
    """Grupo de un docente. Los estudiantes se unen con el código de acceso."""
    __tablename__ = "clases"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, nullable=False)
    descripcion = Column(String, nullable=True)
    codigo = Column(String, unique=True, index=True, nullable=False)
    docente_id = Column(Integer, index=True, nullable=False)
    fecha_creacion = Column(DateTime, default=ahora_utc)

class ClaseEstudiante(Base):
    """Membresía: el estudiante pertenece a la clase."""
    __tablename__ = "clase_estudiantes"
    __table_args__ = (UniqueConstraint("clase_id", "estudiante_id"),)

    id = Column(Integer, primary_key=True)
    clase_id = Column(Integer, index=True, nullable=False)
    estudiante_id = Column(Integer, index=True, nullable=False)
    fecha_union = Column(DateTime, default=ahora_utc)

class ClaseCuestionario(Base):
    """Asignación: el cuestionario está disponible para la clase (si además está publicado)."""
    __tablename__ = "clase_cuestionarios"
    __table_args__ = (UniqueConstraint("clase_id", "cuestionario_id"),)

    id = Column(Integer, primary_key=True)
    clase_id = Column(Integer, index=True, nullable=False)
    cuestionario_id = Column(Integer, index=True, nullable=False)
    fecha_asignacion = Column(DateTime, default=ahora_utc)

# Crear las tablas en Neon (si no existen)
Base.metadata.create_all(bind=engine)

# create_all no agrega columnas nuevas a tablas que ya existen: se agregan aquí.
# (Solo columnas opcionales; cada una se agrega una vez y después se omite.)
COLUMNAS_NUEVAS = {
    Usuario.__tablename__: {"complejidad_preferida": "VARCHAR"},
    Cuestionario.__tablename__: {"complejidad": "VARCHAR"},
    Intento.__tablename__: {"seguridad": "JSON"},
}

def agregar_columnas_faltantes(motor=engine):
    """`motor` se puede cambiar en las pruebas para migrar otra base."""
    inspector = inspect(motor)
    with motor.begin() as conexion:
        for tabla, columnas in COLUMNAS_NUEVAS.items():
            if not inspector.has_table(tabla):
                continue  # create_all la crea completa
            existentes = {c["name"] for c in inspector.get_columns(tabla)}
            for nombre, tipo in columnas.items():
                if nombre not in existentes:
                    conexion.execute(text(f"ALTER TABLE {tabla} ADD COLUMN {nombre} {tipo}"))
                    # Sin emojis: en la consola de Windows (cp1252) el print fallaría y abortaría el arranque
                    print(f"[migracion] Columna agregada: {tabla}.{nombre}")

agregar_columnas_faltantes()