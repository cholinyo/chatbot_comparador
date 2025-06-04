# app/routes/chat.py
from flask import Blueprint, render_template, request
from app.services.rag_context import recuperar_contexto
from app.config.settings import cargar_config

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["GET", "POST"])
def chat():
    respuesta = None
    contexto = []
    pregunta = ""

    if request.method == "POST":
        pregunta = request.form.get("pregunta")
        config = cargar_config()
        k = config.get("rag_k", 3)

        # Recuperar contexto desde documentos
        contexto = recuperar_contexto(pregunta, k=k, fuente="documentos")

        # Simular respuesta concatenando fragmentos
        respuesta = "\n\n".join([f"- {c['texto']}" for c in contexto])

    return render_template("chat.html", pregunta=pregunta, contexto=contexto, respuesta=respuesta)