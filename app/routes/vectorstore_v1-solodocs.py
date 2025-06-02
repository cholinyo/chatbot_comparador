from flask import Blueprint, render_template
import os
import pickle
import faiss
import datetime
import numpy as np
from collections import defaultdict

vectorstore_bp = Blueprint("vectorstore", __name__)
VECTOR_DIR = "vectorstore"
LOG_PATH = os.path.join("logs", "ingestion.log")

@vectorstore_bp.route("/vectorstore")
def vista_vectorstore():
    resumen = {}
    documentos = []
    ejemplo_embedding = None
    log = ""

    # 📥 Cargar metadatos y agrupar por documento
    metadatos = []
    try:
        with open(os.path.join(VECTOR_DIR, "metadatos.pkl"), "rb") as f:
            metadatos = pickle.load(f)

        conteo_documentos = defaultdict(lambda: {"fragmentos": 0, "origen": "documento"})

        for doc in metadatos:
            nombre = doc.get("nombre", "¿?")
            conteo_documentos[nombre]["fragmentos"] += 1
            conteo_documentos[nombre]["origen"] = doc.get("origen", "documento")

        documentos = [
            {"nombre": nombre, "fragmentos": datos["fragmentos"], "origen": datos["origen"]}
            for nombre, datos in conteo_documentos.items()
        ]
    except Exception:
        pass

    # 📈 Cargar índice FAISS
    try:
        index_path = os.path.join(VECTOR_DIR, "index.faiss")
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            resumen["total_fragmentos"] = index.ntotal
            resumen["dimensiones"] = index.d
            resumen["fecha"] = datetime.datetime.fromtimestamp(os.path.getmtime(index_path)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        resumen = None

    # 🧠 Mostrar ejemplo de vector (si está disponible)
    try:
        with open(os.path.join(VECTOR_DIR, "embeddings.npy"), "rb") as f:
            embeddings = np.load(f)
            if len(embeddings) > 0:
                ejemplo_embedding = embeddings[0].tolist()
    except Exception:
        pass

    # 📄 Cargar log de ingesta
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            log = f.read()

    return render_template("vectorstore.html",
                           resumen=resumen,
                           documentos=documentos,
                           ejemplo_embedding=ejemplo_embedding,
                           log=log)
