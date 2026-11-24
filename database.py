import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Boolean, Date, Float, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

# 1. Cargar la URL de tu archivo .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("⚠️ No se encontró DATABASE_URL en el archivo .env")

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
    fecha_creacion = Column(DateTime, default=datetime.datetime.utcnow)
    usuario_id = Column(Integer, index=True)

class Usuario(Base):
    __tablename__ = "usuarios"

    creditos_disponibles = Column(Integer, default=5)
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, index=True)
    correo = Column(String, unique=True, index=True) 
    password_hash = Column(String) 
    rol = Column(String) # 'docente' o 'estudiante'
    fecha_registro = Column(DateTime, default=datetime.datetime.utcnow)
    creditos_disponibles = Column(Integer, default=5)
    fecha_ultimo_uso = Column(Date, default=datetime.date.today)

    foto_perfil = Column(String, nullable=True)
    biografia = Column(String, nullable=True)
    institucion = Column(String, nullable=True)
    especialidad = Column(String, nullable=True)

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
    fecha = Column(DateTime, default=datetime.datetime.utcnow)

class Clase(Base):
    """Grupo de un docente. Los estudiantes se unen con el código de acceso."""
    __tablename__ = "clases"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, nullable=False)
    descripcion = Column(String, nullable=True)
    codigo = Column(String, unique=True, index=True, nullable=False)
    docente_id = Column(Integer, index=True, nullable=False)
    fecha_creacion = Column(DateTime, default=datetime.datetime.utcnow)

class ClaseEstudiante(Base):
    """Membresía: el estudiante pertenece a la clase."""
    __tablename__ = "clase_estudiantes"
    __table_args__ = (UniqueConstraint("clase_id", "estudiante_id"),)

    id = Column(Integer, primary_key=True)
    clase_id = Column(Integer, index=True, nullable=False)
    estudiante_id = Column(Integer, index=True, nullable=False)
    fecha_union = Column(DateTime, default=datetime.datetime.utcnow)

class ClaseCuestionario(Base):
    """Asignación: el cuestionario está disponible para la clase (si además está publicado)."""
    __tablename__ = "clase_cuestionarios"
    __table_args__ = (UniqueConstraint("clase_id", "cuestionario_id"),)

    id = Column(Integer, primary_key=True)
    clase_id = Column(Integer, index=True, nullable=False)
    cuestionario_id = Column(Integer, index=True, nullable=False)
    fecha_asignacion = Column(DateTime, default=datetime.datetime.utcnow)

# Crear las tablas en Neon (si no existen)
Base.metadata.create_all(bind=engine)