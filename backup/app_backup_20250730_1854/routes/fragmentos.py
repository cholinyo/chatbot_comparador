from flask import Blueprint, render_template
import os
import pickle
import urllib.parse

fragmentos_bp = Blueprint("fragmentos", __name__)
VECTOR_DIR = "vectorstore"

@fragmentos_bp.route("/vectorstore/documento/<nombre>")
def ver_fragmentos(nombre):
    nombre_decodificado = urllib.parse.unquote(nombre)
    fragmentos_doc = []
    total = 0

    try:
        # Cargar fragmentos y metadatos
        with open(os.path.join(VECTOR_DIR, "fragmentos.pkl"), "rb") as f:
            todos_fragmentos = pickle.load(f)

        with open(os.path.join(VECTOR_DIR, "metadatos.pkl"), "rb") as f:
            metadatos = pickle.load(f)

        for frag, meta in zip(todos_fragmentos, metadatos):
            if meta.get("nombre") == nombre_decodificado:
                fragmentos_doc.append(frag)

        total = len(fragmentos_doc)

    except Exception as e:
        print(f"⚠️ Error al cargar fragmentos: {e}")

    return render_template("fragmentos_documento.html",
                           nombre_documento=nombre_decodificado,
                           fragmentos=fragmentos_doc,
                           total=total)
