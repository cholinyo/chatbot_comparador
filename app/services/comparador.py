from app.services.bot_local import get_local_response
from app.services.bot_openai import get_openai_response
from app.services.rag_context import recuperar_contexto

import json
import os

# Ruta del archivo settings.json
CONFIG_PATH = os.path.join("app", "config", "settings.json")

def cargar_k_desde_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
            return int(config.get("rag_k", 3))
    except Exception:
        return 3  # Valor por defecto

def comparar_local_vs_openai(pregunta_usuario):
    # Leer valor dinámico de k desde settings.json
    k = cargar_k_desde_config()

    # Recuperar contexto (RAG)
    contexto = recuperar_contexto(pregunta_usuario, k=k)

    # Construir bloque de contexto para el prompt
    contexto_textual = "\n\n".join(
        f"[{i+1}] {frag['texto']}" for i, frag in enumerate(contexto)
    )

    # Prompt con contexto incluido
    prompt_con_contexto = (
        f"Contexto recuperado:\n{contexto_textual}\n\n"
        f"Pregunta: {pregunta_usuario}"
    )

    # Obtener respuestas de ambos modelos
    respuesta_local = get_local_response(prompt_con_contexto)
    respuesta_openai = get_openai_response(prompt_con_contexto)

    return {
        "pregunta": pregunta_usuario,
        "contexto": contexto_textual,
        "respuesta_local": respuesta_local,
        "respuesta_openai": respuesta_openai
    }
