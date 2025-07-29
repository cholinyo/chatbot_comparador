from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print("🔐 Clave cargada:", api_key[:10], "...")

client = OpenAI(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hola, ¿qué tal?"}]
    )
    print("✅ Respuesta:", response.choices[0].message.content)
except Exception as e:
    print("❌ Error:", e)
