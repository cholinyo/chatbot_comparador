import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_response(prompt_usuario):
    from app.config.settings import get_openai_model

    try:
        system_prompt = (
            "Eres un asistente experto de OpenAI. "
            "Responde con precisión y explica claramente tus respuestas."
        )

        response = openai.ChatCompletion.create(
            model=get_openai_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_usuario}
            ],
            max_tokens=512,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Error OpenAI: {e}"
