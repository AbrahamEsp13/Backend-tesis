import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from google import genai
from google.genai import types
import bcrypt
from pydantic import BaseModel
from database import Usuario # Asegúrate de importar tu nueva tabla

# --- NUEVAS IMPORTACIONES DE BASE DE DATOS ---
from sqlalchemy.orm import Session
from database import SessionLocal, Cuestionario 

# Modelos para recibir datos desde React
class RegistroUsuario(BaseModel):
    nombre: str
    correo: str
    password: str
    rol: str

class LoginUsuario(BaseModel):
    correo: str
    password: str

class ActualizarCuestionario(BaseModel):
    preguntas_json: list

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
chroma_client = chromadb.Client()
coleccion = chroma_client.get_or_create_collection(name="tesis_api_cuestionarios")

app = FastAPI(title="API Backend Tesis E-Learning", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependencia para abrir y cerrar la conexión a la base de datos en cada petición
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/generar-cuestionario")
async def generar_cuestionario(
    archivo: UploadFile = File(...), 
    usuario_id: int = Form(...), # <-- AQUÍ RECIBIMOS AL DUEÑO
    db: Session = Depends(get_db)
):
    if not archivo.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF")
    
    ruta_temp = f"temp_{archivo.filename}"
    with open(ruta_temp, "wb") as buffer:
        buffer.write(await archivo.read())
        
    try:
        lector = PdfReader(ruta_temp)
        texto_completo = ""
        for pagina in lector.pages:
            texto_completo += pagina.extract_text() + "\n"
            
        tamano_fragmento = 1000
        fragmentos = [texto_completo[i:i+tamano_fragmento] for i in range(0, len(texto_completo), tamano_fragmento)]
        
        coleccion.add(
            documents=fragmentos,
            ids=[f"frag_{archivo.filename}_{i}" for i in range(len(fragmentos))]
        )
        
        resultados = coleccion.query(query_texts=["Genera preguntas de opción múltiple"], n_results=2)
        contexto_recuperado = "\n".join(resultados['documents'][0])
        
        prompt_final = f"""
        Eres un profesor experto en pedagogía. Genera una evaluación basada ÚNICAMENTE en el contexto proporcionado. No inventes información.
        
        Genera exactamente 3 preguntas de opción múltiple. Cada pregunta debe corresponder a un nivel cognitivo diferente de la Taxonomía de Bloom:
        1. Recordar
        2. Comprender
        3. Analizar
        
        El formato de salida DEBE ser un arreglo JSON con la siguiente estructura:
        [
          {{
            "nivel_bloom": "Recordar",
            "pregunta": "¿...",
            "opciones": ["a) ...", "b) ...", "c) ...", "d) ..."],
            "respuesta_correcta": "La opción correcta exacta",
            "justificacion_pedagogica": "Por qué es correcta basada en el texto"
          }}
        ]
        
        Contexto del PDF:
        {contexto_recuperado}
        """
        
        respuesta = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_final,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2 
            )
        )
        
        cuestionario_json = json.loads(respuesta.text)
        
        # --- NUEVA LÓGICA CON REINTENTOS PARA NEON SERVERLESS ---
        import time # <-- Asegúrate de que esto esté aquí o hasta arriba de tu archivo
        max_reintentos = 3
        
        for intento in range(max_reintentos):
            try:
                print(f"💾 Intentando guardar en Neon (Intento {intento + 1})...")
                nuevo_registro = Cuestionario(
                    nombre_documento=archivo.filename,
                    preguntas_json=cuestionario_json,
                    usuario_id=usuario_id
                )
                db.add(nuevo_registro)
                db.commit()
                db.refresh(nuevo_registro)
                break # Si tiene éxito, rompemos el ciclo de reintentos
            except Exception as e:
                # Si falla por la conexión SSL, cancelamos el intento fallido
                db.rollback() 
                if intento < max_reintentos - 1:
                    print(f"⚠️ Neon estaba dormido o cortó la conexión. Reintentando en 2 segundos...")
                    time.sleep(2) # Esperamos a que Neon despierte
                else:
                    print(f"❌ Error crítico en base de datos tras 3 intentos: {e}")
                    raise HTTPException(status_code=500, detail="Error de conexión con la base de datos. Por favor, intenta de nuevo.")
        # ---------------------------------------------
        
    except Exception as e:
        print(f"❌ Error interno: {e}")
        raise HTTPException(status_code=500, detail="Ocurrió un error al procesar el documento.")
        
    finally:
        if os.path.exists(ruta_temp):
            os.remove(ruta_temp)

@app.get("/api/cuestionarios")
def obtener_historial(usuario_id: int = None, rol: str = None, db: Session = Depends(get_db)):
    print("📚 Consultando el historial filtrado...")
    try:
        if rol == 'estudiante':
            # Estudiantes ven TODOS los publicados
            historial = db.query(Cuestionario).filter(Cuestionario.publicado == True).order_by(Cuestionario.id.desc()).all()
        elif rol == 'docente' and usuario_id:
            # Docentes ven SOLO los suyos
            historial = db.query(Cuestionario).filter(Cuestionario.usuario_id == usuario_id).order_by(Cuestionario.id.desc()).all()
        else:
            historial = []
            
        return {"data": historial}
    except Exception as e:
        print(f"❌ Error al consultar base de datos: {e}")
        raise HTTPException(status_code=500, detail="Error al obtener el historial.")


@app.put("/api/cuestionarios/{cuestionario_id}/publicar")
def publicar_cuestionario(cuestionario_id: int, db: Session = Depends(get_db)):
    # Buscamos el cuestionario en la base de datos
    cuestionario = db.query(Cuestionario).filter(Cuestionario.id == cuestionario_id).first()
    
    if not cuestionario:
        raise HTTPException(status_code=404, detail="Cuestionario no encontrado")
    
    # Cambiamos el interruptor a True
    cuestionario.publicado = True
    db.commit()
    return {"mensaje": "Cuestionario publicado exitosamente"}


@app.delete("/api/cuestionarios/{cuestionario_id}")
def eliminar_cuestionario(cuestionario_id: int, db: Session = Depends(get_db)):
    print(f"🗑️ Solicitud para eliminar el cuestionario ID: {cuestionario_id}")
    
    # 1. Buscamos el cuestionario
    cuestionario = db.query(Cuestionario).filter(Cuestionario.id == cuestionario_id).first()
    
    if not cuestionario:
        raise HTTPException(status_code=404, detail="Cuestionario no encontrado")
    
    # 2. Lo eliminamos de la base de datos
    db.delete(cuestionario)
    db.commit()
    
    return {"mensaje": "Cuestionario eliminado exitosamente"}

@app.put("/api/cuestionarios/{id}")
def actualizar_cuestionario(id: int, datos: ActualizarCuestionario, db: Session = Depends(get_db)):
    # 1. Buscamos el cuestionario en la base de datos
    cuestionario = db.query(Cuestionario).filter(Cuestionario.id == id).first()
    
    if not cuestionario:
        raise HTTPException(status_code=404, detail="Cuestionario no encontrado")
    
    # 2. Reemplazamos el JSON viejo con el nuevo JSON editado
    cuestionario.preguntas_json = datos.preguntas_json
    
    # 3. Guardamos los cambios
    db.commit()
    
    return {"mensaje": "Cuestionario actualizado correctamente"}

# ==========================================
# RUTAS DE AUTENTICACIÓN (LOGIN Y REGISTRO)
# ==========================================

@app.post("/api/auth/register")
def registrar_usuario(datos: RegistroUsuario, db: Session = Depends(get_db)):
    # 1. Verificar si el correo ya existe
    usuario_existente = db.query(Usuario).filter(Usuario.correo == datos.correo).first()
    if usuario_existente:
        raise HTTPException(status_code=400, detail="El correo ya está registrado")
    
    # 2. Encriptar la contraseña (forma nativa y moderna)
    salt = bcrypt.gensalt()
    password_encriptada = bcrypt.hashpw(datos.password.encode('utf-8'), salt).decode('utf-8')
    
    # 3. Guardar en la base de datos
    nuevo_usuario = Usuario(
        nombre=datos.nombre,
        correo=datos.correo,
        password_hash=password_encriptada,
        rol=datos.rol
    )
    db.add(nuevo_usuario)
    db.commit()
    db.refresh(nuevo_usuario)
    
    return {"mensaje": "Usuario registrado exitosamente", "rol": nuevo_usuario.rol}

@app.post("/api/auth/login")
def iniciar_sesion(datos: LoginUsuario, db: Session = Depends(get_db)):
    # 1. Buscar al usuario por su correo
    usuario = db.query(Usuario).filter(Usuario.correo == datos.correo).first()
    if not usuario:
        raise HTTPException(status_code=404, detail="Correo no encontrado")
    
    # 2. Verificar que la contraseña coincida con el hash (usando bcrypt nativo)
    if not bcrypt.checkpw(datos.password.encode('utf-8'), usuario.password_hash.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Contraseña incorrecta")
    
    # 3. ¡Todo correcto! Devolvemos los datos del usuario
    return {
        "mensaje": "Login exitoso", 
        "usuario": {
            "id": usuario.id,
            "nombre": usuario.nombre,
            "correo": usuario.correo,
            "rol": usuario.rol
        }
    }