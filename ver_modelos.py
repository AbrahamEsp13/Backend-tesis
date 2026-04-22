import os
from dotenv import load_dotenv
from google import genai

# Cargar la llave
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ No se encontró la API Key en el archivo .env")
    exit()

# Inicializar cliente
client = genai.Client(api_key=api_key)

print("🔍 Preguntándole a Google qué modelos tienes disponibles...")
print("-" * 50)

try:
    # Obtenemos la lista y solo imprimimos el nombre
    for model in client.models.list():
        # Filtramos para ver solo los más relevantes para tu tesis
        if "flash" in model.name or "pro" in model.name:
            print(f"✅ {model.name}")
except Exception as e:
    print(f"❌ Ocurrió un error al consultar: {e}")