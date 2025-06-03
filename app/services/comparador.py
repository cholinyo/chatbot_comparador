
from app.utils.rag_utils import buscar_fragmentos
from app.services.bot_openai import get_openai_response
from app.services.bot_local import get_local_response

def construir_prompt_con_fragmentos(pregunta, fragmentos):
    contexto = "\n".join([f"- {f.get('texto', '')}" for f in fragmentos])
    prompt = f"""Usa la siguiente información para responder a la pregunta:

{contexto}

Pregunta: {pregunta}
Respuesta:"""
    return prompt

def comparar_local_vs_openai(pregunta):
    fragmentos = buscar_fragmentos(pregunta, k=3)
    prompt = construir_prompt_con_fragmentos(pregunta, fragmentos)

    respuesta_openai = get_openai_response(prompt)
    respuesta_local = get_local_response(prompt)

    return {
        "pregunta": pregunta,
        "respuesta_openai": respuesta_openai,
        "respuesta_local": respuesta_local,
        "fragmentos": fragmentos
    }
