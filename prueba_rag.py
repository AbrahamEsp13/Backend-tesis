import os
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from google import genai
from google.genai import types

# 1. Cargar las credenciales ocultas
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("⚠️ Error: No se encontró la GEMINI_API_KEY en el archivo .env")

# 2. Inicializar los clientes
# El nuevo SDK inicializa el cliente de forma diferente
client = genai.Client(api_key=GEMINI_API_KEY)

# Inicia ChromaDB en memoria
chroma_client = chromadb.Client() 

# Crear nuestra colección 
coleccion = chroma_client.create_collection(name="tesis_documentos_gemini_v2")

def extraer_texto_pdf(ruta_archivo):
    """Extrae todo el texto de un PDF."""
    print(f"📄 Leyendo el archivo: {ruta_archivo}...")
    lector = PdfReader(ruta_archivo)
    texto = ""
    for pagina in lector.pages:
        texto += pagina.extract_text() + "\n"
    return texto

def ejecutar_prueba_rag(ruta_pdf):
    # PASO A: Procesar el documento
    texto_completo = extraer_texto_pdf(ruta_pdf)

    print("✂️ Dividiendo el texto en fragmentos...")
    tamano_fragmento = 1000
    fragmentos = [texto_completo[i:i+tamano_fragmento] for i in range(0, len(texto_completo), tamano_fragmento)]

    # PASO B: Guardar en la base de datos vectorial
    print("🧠 Generando embeddings locales y guardando en ChromaDB...")
    coleccion.add(
        documents=fragmentos,
        ids=[f"frag_{i}" for i in range(len(fragmentos))]
    )

    # PASO C: Recuperación (Retrieval)
    instruccion_docente = "Genera 3 preguntas de opción múltiple."
    print(f"🔍 Buscando contexto relevante en la base de datos vectorial...")
    
    resultados = coleccion.query(
        query_texts=[instruccion_docente],
        n_results=2 
    )
    
    contexto_recuperado = "\n".join(resultados['documents'][0])

    # PASO D: Generación Aumentada con Gemini (Estructurada)
    print("🤖 Solicitando el cuestionario estructurado a Gemini 2.5 Flash...")
    
    prompt_final = f"""
    Eres un profesor experto en pedagogía. Tu tarea es generar una evaluación basada ÚNICAMENTE en el contexto proporcionado. No inventes información.
    
    Genera exactamente 3 preguntas de opción múltiple. Cada pregunta debe corresponder a un nivel cognitivo diferente de la Taxonomía de Bloom:
    1. Recordar
    2. Comprender
    3. Analizar
    
    El formato de salida DEBE ser un arreglo JSON con la siguiente estructura exacta para cada pregunta:
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
    
    # Le indicamos a Gemini que su respuesta DEBE ser un JSON válido
    respuesta = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt_final,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2 # Temperatura muy baja para máxima precisión
        )
    )

    print("\n" + "="*60)
    print("🎯 JSON GENERADO POR IA LISTO PARA TU API:")
    print("="*60)
    print(respuesta.text)

if __name__ == "__main__":
    # Asegúrate de que el nombre coincida con tu PDF
    NOMBRE_DEL_PDF = "documento.pdf" 
    
    if os.path.exists(NOMBRE_DEL_PDF):
        ejecutar_prueba_rag(NOMBRE_DEL_PDF)
    else:
        print(f"❌ Error: No se encontró el archivo '{NOMBRE_DEL_PDF}' en esta carpeta.")