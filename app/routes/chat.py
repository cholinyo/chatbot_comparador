
from flask import Blueprint, render_template, request
from app.services.comparador import comparar_local_vs_openai

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/comparar", methods=["GET", "POST"])
def comparar():
    if request.method == "POST":
        pregunta = request.form.get("pregunta", "").strip()
        if not pregunta:
            return render_template("comparar.html", error="⚠️ Introduce una pregunta válida.")
        resultado = comparar_local_vs_openai(pregunta)
        return render_template("comparar.html", **resultado)
    return render_template("comparar.html")
