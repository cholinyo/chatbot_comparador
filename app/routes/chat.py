from flask import Blueprint, render_template, request
from app.utils.rag_utils import buscar_fragmentos_combinados
from app.services.bot_openai import get_openai_response
from app.services.bot_local import get_local_response

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["GET", "POST"])
def chat():
    respuesta = None
    fragmentos = []
    pregunta = ""

    if request.method == "POST":
        pregunta = request.form.get("pregunta")
        fragmentos = buscar_fragmentos_combinados(pregunta, k=5)

        contexto = "\n".join([f"- {f['texto']}" for f in fragmentos])
        prompt = f"""Usa la siguiente información para responder a la pregunta:

{contexto}

Pregunta: {pregunta}
Respuesta:"""

        respuesta = get_local_response(prompt)

    return render_template("chat.html", pregunta=pregunta, respuesta=respuesta, contexto=fragmentos)