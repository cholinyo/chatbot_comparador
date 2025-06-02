from flask import Blueprint, render_template

grafo_bp = Blueprint("grafo", __name__)

@grafo_bp.route("/grafo")
def ver_grafo():
    # Página temporal de placeholder
    return render_template("grafo.html")
