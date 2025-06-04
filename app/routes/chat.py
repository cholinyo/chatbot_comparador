# app/routes/chat.py (unificando fuentes: documentos + web con relevancia y fuente visible)
from flask import Blueprint, render_template, request
from app.services.rag_context import recuperar_contexto
from app.config.settings import cargar_config

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["GET", "POST"])
def chat():
    respuesta = None
    contexto = []
    pregunta = ""
    origen = "ambas"  # Unifica documentos + web

    if request.method == "POST":
        pregunta = request.form.get("pregunta")
        config = cargar_config()
        k = config.get("rag_k", 3)

        # Recuperar contexto desde ambas fuentes
        contexto = recuperar_contexto(pregunta, k=k, fuente="ambas")

        # Ordenar por relevancia (distancia más baja primero)
        contexto.sort(key=lambda x: x.get("distancia", 0))

        # Simular respuesta concatenando fragmentos con fuente
        respuesta = "\n\n".join([
            f"[{c.get('fuente', '¿?')} | {c.get('distancia', '?'):.3f}] {c['texto']}" for c in contexto
        ])

    return render_template(
        "chat.html",
        pregunta=pregunta,
        contexto=contexto,
        respuesta=respuesta,
        origen=origen
    )