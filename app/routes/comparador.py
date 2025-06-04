# app/routes/comparador.py
from flask import Blueprint, render_template, request
from app.utils.rag_utils import buscar_fragmentos
from app.services.bot_openai import get_openai_response
from app.services.bot_local import get_local_response
from app.config.settings import cargar_config
import time

comparador_bp = Blueprint("comparador", __name__)

@comparador_bp.route("/comparar", methods=["GET", "POST"])
def comparar():
    respuesta_openai = None
    respuesta_local = None
    tiempo_openai = None
    tiempo_local = None
    fragmentos = []
    pregunta = ""

    if request.method == "POST":
        pregunta = request.form.get("pregunta")
        config = cargar_config()
        k = config.get("rag_k", 3)
        fragmentos = buscar_fragmentos(pregunta, k=k)

        contexto = "\n".join([f"- {f.get('fragmento') or f.get('texto', '')}" for f in fragmentos])
        prompt = f"""Usa la siguiente información para responder a la pregunta:

{contexto}

Pregunta: {pregunta}
Respuesta:"""

        try:
            inicio = time.time()
            respuesta_openai = get_openai_response(prompt)
            tiempo_openai = round(time.time() - inicio, 2)
        except Exception as e:
            respuesta_openai = f"[ERROR OpenAI] {str(e)}"
            tiempo_openai = None

        try:
            inicio = time.time()
            respuesta_local = get_local_response(prompt)
            tiempo_local = round(time.time() - inicio, 2)
        except Exception as e:
            respuesta_local = f"[ERROR Modelo Local] {str(e)}"
            tiempo_local = None

    return render_template("comparar.html", 
                           pregunta=pregunta, 
                           respuesta_openai=respuesta_openai, 
                           respuesta_local=respuesta_local, 
                           tiempo_openai=tiempo_openai, 
                           tiempo_local=tiempo_local, 
                           fragmentos=fragmentos) 
