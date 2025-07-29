# app/routes/comparador.py

from flask import Blueprint, render_template, request
import time
from app.utils.rag_utils import buscar_fragmentos_combinados
from app.services.bot_openai import get_openai_response
from app.services.bot_local import get_local_response

comparador_bp = Blueprint("comparador", __name__)

@comparador_bp.route("/comparar", methods=["GET", "POST"])
def comparar():
    respuesta_openai = None
    respuesta_local = None
    tiempo_openai = "-"
    tiempo_local = "-"
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

        # OpenAI
        start_openai = time.time()
        respuesta_openai = get_openai_response(prompt)
        end_openai = time.time()
        if not respuesta_openai.startswith("⚠️"):
            tiempo_openai = round(end_openai - start_openai, 2)

        # Local
        start_local = time.time()
        respuesta_local = get_local_response(prompt)
        end_local = time.time()
        if not respuesta_local.startswith("⚠️"):
            tiempo_local = round(end_local - start_local, 2)

    return render_template("comparar.html",
                           pregunta=pregunta,
                           respuesta_openai=respuesta_openai,
                           respuesta_local=respuesta_local,
                           fragmentos=fragmentos,
                           tiempo_openai=tiempo_openai,
                           tiempo_local=tiempo_local)
