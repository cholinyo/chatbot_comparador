from flask import Blueprint, render_template, request
from app.utils.rag_utils import buscar_fragmentos_combinados
from app.services.bot_openai import get_openai_response
from app.services.bot_local import get_local_response

comparador_bp = Blueprint("comparador", __name__)

@comparador_bp.route("/comparar", methods=["GET", "POST"])
def comparar():
    respuesta_openai = None
    respuesta_local = None
    fragmentos = []
    pregunta = ""

    if request.method == "POST":
        pregunta = request.form.get("pregunta")
        fragmentos = buscar_fragmentos_combinados(pregunta, k=3)

        contexto = "\n".join([f"- {f['texto']}" for f in fragmentos])
        prompt = f"""Usa la siguiente información para responder a la pregunta:

{contexto}

Pregunta: {pregunta}
Respuesta:"""

        respuesta_openai = get_openai_response(prompt)
        respuesta_local = get_local_response(prompt)

    return render_template("comparar.html",
                           pregunta=pregunta,
                           respuesta_openai=respuesta_openai,
                           respuesta_local=respuesta_local,
                           fragmentos=fragmentos)