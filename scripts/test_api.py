from dotenv import load_dotenv
import os
import openai

load_dotenv()

print("🔎 Versión de openai instalada:", openai.__version__)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ No se ha encontrado la clave OPENAI_API_KEY.")
    exit()

print("🔐 Clave cargada:", api_key[:15], "...")

try:
    client = openai.OpenAI(api_key=api_key)  # solo válido para openai>=1.0.0
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "¿Cuál es la capital de España?"}]
    )
    print("✅ Conexión exitosa. Respuesta:", response.choices[0].message.content)
except Exception as e:
    print("❌ Error al conectar:", e)
