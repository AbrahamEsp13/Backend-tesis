import os
import json
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from google import genai
from google.genai import types
import bcrypt
from pydantic import BaseModel

# --- IMPORTACIONES DE BASE DE DATOS ---
from sqlalchemy.orm import Session
from database import SessionLocal, Cuestionario, Usuario

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

class ClonarRequest(BaseModel):
    usuario_id: int # ID del docente que está clonando el examen

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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/generar-cuestionario")
async def generar_cuestionario(
    archivo: UploadFile = File(...), 
    usuario_id: int = Form(...), 
    num_preguntas: int = Form(5),
    nombre_examen: str = Form(...), 
    materia: str = Form(...),
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
        
        Genera exactamente {num_preguntas} preguntas de opción múltiple. Las preguntas deben estar distribuidas en los diferentes niveles cognitivos de la Taxonomía de Bloom (Recordar, Comprender, Aplicar, Analizar, Evaluar, Crear).
        
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
        
        max_reintentos = 3
        for intento in range(max_reintentos):
            try:
                print(f"💾 Intentando guardar en Neon (Intento {intento + 1})...")
                nuevo_registro = Cuestionario(
                    nombre_examen=nombre_examen,      
                    nombre_documento=archivo.filename,
                    materia=materia,
                    preguntas_json=cuestionario_json, 
                    usuario_id=usuario_id,            
                    compartido_comunidad=False
                )
                db.add(nuevo_registro)
                db.commit()
                db.refresh(nuevo_registro)
                break 
            except Exception as e:
                db.rollback() 
                if intento < max_reintentos - 1:
                    print(f"⚠️ Neon estaba dormido o cortó la conexión. Reintentando en 2 segundos...")
                    time.sleep(2) 
                else:
                    print(f"❌ Error crítico en base de datos tras 3 intentos: {e}")
                    raise HTTPException(status_code=500, detail="Error de conexión con la base de datos.")

        usuario.creditos_disponibles -= 1
        db.commit()
        return {"status": "success", "mensaje": "Cuestionario generado correctamente"}
                    
    except Exception as e:
        print(f"❌ Error interno: {e}")
        raise HTTPException(status_code=500, detail="Ocurrió un error al procesar el documento.")
        
    finally:
        if os.path.exists(ruta_temp):
            os.remove(ruta_temp)

@app.get("/api/cuestionarios")
def obtener_historial(usuario_id: int = None, rol: str = None, db: Session = Depends(get_db)):
    try:
        if rol == 'estudiante':
            historial = db.query(Cuestionario).filter(Cuestionario.publicado == True).order_by(Cuestionario.id.desc()).all()
        elif rol == 'docente' and usuario_id:
            historial = db.query(Cuestionario).filter(Cuestionario.usuario_id == usuario_id).order_by(Cuestionario.id.desc()).all()
        else:
            historial = []
            
        return {"data": historial}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al obtener el historial.")

@app.put("/api/cuestionarios/{cuestionario_id}/publicar")
def publicar_cuestionario(cuestionario_id: int, db: Session = Depends(get_db)):
    cuestionario = db.query(Cuestionario).filter(Cuestionario.id == cuestionario_id).first()
    if not cuestionario:
        raise HTTPException(status_code=404, detail="Cuestionario no encontrado")
    
    cuestionario.publicado = True
    db.commit()
    return {"mensaje": "Cuestionario publicado exitosamente"}

@app.put("/api/cuestionarios/{cuestionario_id}/despublicar")
def despublicar_cuestionario(cuestionario_id: int, db: Session = Depends(get_db)):
    cuestionario = db.query(Cuestionario).filter(Cuestionario.id == cuestionario_id).first()
    if not cuestionario:
        raise HTTPException(status_code=404, detail="Cuestionario no encontrado")
    
    cuestionario.publicado = False
    db.commit()
    return {"mensaje": "Cuestionario ocultado exitosamente"}

@app.delete("/api/cuestionarios/{cuestionario_id}")
def eliminar_cuestionario(cuestionario_id: int, db: Session = Depends(get_db)):
    cuestionario = db.query(Cuestionario).filter(Cuestionario.id == cuestionario_id).first()
    if not cuestionario:
        raise HTTPException(status_code=404, detail="Cuestionario no encontrado")
    
    db.delete(cuestionario)
    db.commit()
    return {"mensaje": "Cuestionario eliminado exitosamente"}

@app.put("/api/cuestionarios/{id}")
def actualizar_cuestionario(id: int, datos: ActualizarCuestionario, db: Session = Depends(get_db)):
    cuestionario = db.query(Cuestionario).filter(Cuestionario.id == id).first()
    if not cuestionario:
        raise HTTPException(status_code=404, detail="Cuestionario no encontrado")
    
    cuestionario.preguntas_json = datos.preguntas_json
    db.commit()
    return {"mensaje": "Cuestionario actualizado correctamente"}


# ==========================================================
# NUEVOS ENDPOINTS: MERCADO COMUNITARIO DE DOCENTES
# ==========================================================

@app.get("/api/comunidad/cuestionarios")
def obtener_cuestionarios_comunidad(db: Session = Depends(get_db)):
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
        print(f"❌ Error en Comunidad: {e}")
        raise HTTPException(status_code=500, detail="Error al cargar la comunidad.")

@app.put("/api/cuestionarios/{cuestionario_id}/compartir")
def compartir_en_comunidad(cuestionario_id: int, db: Session = Depends(get_db)):
    """Cambia el interruptor para hacer el examen público en el mercado comunitario"""
    cuestionario = db.query(Cuestionario).filter(Cuestionario.id == cuestionario_id).first()
    if not cuestionario:
        raise HTTPException(status_code=404, detail="Cuestionario no encontrado")
    
    cuestionario.compartido_comunidad = True
    db.commit()
    return {"status": "success", "mensaje": "Examen compartido en la comunidad con éxito"}

@app.post("/api/cuestionarios/{cuestionario_id}/clonar")
def clonar_cuestionario(cuestionario_id: int, datos: ClonarRequest, db: Session = Depends(get_db)):
    """Toma un examen de la comunidad, crea un clon exacto añadiendo '(Copia)' y se lo asigna al nuevo usuario_id"""
    cuestionario_original = db.query(Cuestionario).filter(Cuestionario.id == cuestionario_id).first()
    if not cuestionario_original:
        raise HTTPException(status_code=404, detail="Cuestionario base no encontrado")
        
    try:
        nuevo_clon = Cuestionario(
            nombre_examen=f"{cuestionario_original.nombre_examen} (Copia)",
            nombre_documento=cuestionario_original.nombre_documento,
            materia=cuestionario_original.materia,
            preguntas_json=cuestionario_original.preguntas_json,
            usuario_id=datos.usuario_id, # <--- Se asigna al nuevo docente que lo clonó
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
    usuario_existente = db.query(Usuario).filter(Usuario.correo == datos.correo).first()
    if usuario_existente:
        raise HTTPException(status_code=400, detail="El correo ya está registrado")
    
    salt = bcrypt.gensalt()
    password_encriptada = bcrypt.hashpw(datos.password.encode('utf-8'), salt).decode('utf-8')
    
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
    usuario = db.query(Usuario).filter(Usuario.correo == datos.correo).first()
    if not usuario:
        raise HTTPException(status_code=404, detail="Correo no encontrado")
    
    if not bcrypt.checkpw(datos.password.encode('utf-8'), usuario.password_hash.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Contraseña incorrecta")
    
    return {
        "mensaje": "Login exitoso", 
        "usuario": {
            "id": usuario.id,
            "nombre": usuario.nombre,
            "correo": usuario.correo,
            "rol": usuario.rol,
            "creditos_disponibles": usuario.creditos_disponibles
        }
    }