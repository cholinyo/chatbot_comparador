from flask import Blueprint, render_template, request
from app.services.comparador import comparar_local_vs_openai

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/", methods=["GET", "POST"])
def chat():
    resultado = None

    if request.method == "POST":
        pregunta = request.form.get("pregunta")
        if pregunta:
            resultado = comparar_local_vs_openai(pregunta)

    return render_template("chat.html", resultado=resultado)
