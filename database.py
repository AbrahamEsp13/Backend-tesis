import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

# 1. Cargar la URL de tu archivo .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("⚠️ No se encontró DATABASE_URL en el archivo .env")

# 2. Configurar el motor de conexión
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 3. Definir la Tabla Entidad-Relación
class Cuestionario(Base):
    __tablename__ = "cuestionarios_v2" # <--- NUEVO NOMBRE DE TABLA PARA EVITAR CONFLICTOS

    id = Column(Integer, primary_key=True, index=True)
    nombre_documento = Column(String, index=True)
    preguntas_json = Column(JSON, nullable=False) 
    publicado = Column(Boolean, default=False) # <--- NUEVO INTERRUPTOR (Por defecto en Borrador)
    fecha_creacion = Column(DateTime, default=datetime.datetime.utcnow)
    usuario_id = Column(Integer, index=True)

class Usuario(Base):
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, index=True)
    correo = Column(String, unique=True, index=True) # unique=True evita correos duplicados
    password_hash = Column(String) # Aquí guardaremos la contraseña encriptada
    rol = Column(String) # 'docente' o 'estudiante'
    fecha_registro = Column(DateTime, default=datetime.datetime.utcnow)


# Crear las tablas en Neon (si no existen)
Base.metadata.create_all(bind=engine)



