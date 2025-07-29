# app/services/bot_openai.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from app.config.settings import get_openai_model

# Carga las variables de entorno desde .env
load_dotenv()

# Instancia moderna del cliente OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY no está definida en el entorno o en .env")

client = OpenAI(api_key=OPENAI_API_KEY)

def get_openai_response(prompt_usuario: str) -> str:
    """
    Genera una respuesta usando la API moderna de OpenAI.
    """
    try:
        system_prompt = (
            "Eres un asistente experto de OpenAI. "
            "Responde con precisión y explica claramente tus respuestas."
        )

        completion = client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_usuario}
            ],
            max_tokens=512,
            temperature=0.7
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Error OpenAI: {e}"

