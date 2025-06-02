# app/routes/vectorstore.py (actualizado para reflejar todos los fragmentos correctamente)
from flask import Blueprint, render_template
import os
import pickle
import faiss
import datetime
import numpy as np

vectorstore_bp = Blueprint("vectorstore", __name__)
VECTOR_DIR = "vectorstore"
LOG_PATH_DOC = os.path.join("logs", "ingestion.log")
LOG_PATH_WEB = os.path.join("logs", "ingestion_web.log")

@vectorstore_bp.route("/vectorstore")
def vista_vectorstore():
    resumen = {}
    documentos = []
    urls = []
    ejemplo_embedding = None
    log_docs = ""
    log_web = ""

    # 📥 Cargar metadatos
    metadatos_path = os.path.join(VECTOR_DIR, "metadatos.pkl")
    if os.path.exists(metadatos_path):
        with open(metadatos_path, "rb") as f:
            metadatos = pickle.load(f)
            for doc in metadatos:
                entry = {
                    "nombre": doc.get("nombre", "¿?"),
                    "fragmentos": doc.get("total_fragmentos", "?"),
                    "origen": doc.get("origen", "documento")
                }
                if entry["origen"] == "url":
                    urls.append(entry)
                else:
                    documentos.append(entry)

    # 📈 Cargar índice FAISS
    try:
        index_path = os.path.join(VECTOR_DIR, "index.faiss")
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            resumen["total_fragmentos"] = index.ntotal
            resumen["dimensiones"] = index.d
            resumen["fecha"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(index_path)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        resumen = None

    # 🧠 Mostrar ejemplo de vector
    try:
        with open(os.path.join(VECTOR_DIR, "embeddings.npy"), "rb") as f:
            embeddings = np.load(f)
            if len(embeddings) > 0:
                ejemplo_embedding = embeddings[0].tolist()
    except Exception:
        pass

    # 📄 Cargar logs
    if os.path.exists(LOG_PATH_DOC):
        with open(LOG_PATH_DOC, "r", encoding="utf-8") as f:
            log_docs = f.read()

    if os.path.exists(LOG_PATH_WEB):
        with open(LOG_PATH_WEB, "r", encoding="utf-8") as f:
            log_web = f.read()

    return render_template("vectorstore.html",
                           resumen=resumen,
                           documentos=documentos,
                           urls=urls,
                           ejemplo_embedding=ejemplo_embedding,
                           log_docs=log_docs,
                           log_web=log_web)
