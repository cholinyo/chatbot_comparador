
from flask import Blueprint, render_template, request
import os
import pickle
import faiss
import datetime
import numpy as np

vectorstore_bp = Blueprint("vectorstore", __name__)
BASE_DIR = "vectorstore"
LOG_PATH_DOC = os.path.join("logs", "ingestion_documents.log")
LOG_PATH_WEB = os.path.join("logs", "ingestion_web.log")

def cargar_metadatos_y_resumen(subcarpeta):
    ruta = os.path.join(BASE_DIR, subcarpeta)
    metadatos = []
    resumen = {}

    path_meta = os.path.join(ruta, "metadatos.pkl")
    path_index = os.path.join(ruta, "index.faiss")
    path_embeddings = os.path.join(ruta, "embeddings.npy")

    if os.path.exists(path_meta):
        with open(path_meta, "rb") as f:
            metadatos = pickle.load(f)

    if os.path.exists(path_index):
        index = faiss.read_index(path_index)
        resumen = {
            "total_fragmentos": index.ntotal,
            "dimensiones": index.d,
            "fecha": datetime.datetime.fromtimestamp(os.path.getmtime(path_index)).strftime("%Y-%m-%d %H:%M")
        }
    else:
        resumen = None

    ejemplo_vector = None
    if os.path.exists(path_embeddings):
        try:
            with open(path_embeddings, "rb") as f:
                emb = np.load(f)
                if len(emb) > 0:
                    ejemplo_vector = emb[0].tolist()
        except Exception:
            pass

    return metadatos, resumen, ejemplo_vector

@vectorstore_bp.route("/vectorstore")
def vista_vectorstore():
    documentos, resumen_docs, ejemplo_embedding_docs = cargar_metadatos_y_resumen("documents")
    urls, resumen_web, ejemplo_embedding_web = cargar_metadatos_y_resumen("web")

    log_docs = ""
    log_web = ""
    if os.path.exists(LOG_PATH_DOC):
        with open(LOG_PATH_DOC, "r", encoding="utf-8") as f:
            log_docs = f.read()
    if os.path.exists(LOG_PATH_WEB):
        with open(LOG_PATH_WEB, "r", encoding="utf-8") as f:
            log_web = f.read()

    return render_template("vectorstore.html",
                           resumen_docs=resumen_docs,
                           resumen_web=resumen_web,
                           documentos=documentos,
                           urls=urls,
                           ejemplo_embedding_docs=ejemplo_embedding_docs,
                           ejemplo_embedding_web=ejemplo_embedding_web,
                           log_docs=log_docs,
                           log_web=log_web)

@vectorstore_bp.route("/vectorstore/documento/<nombre>")
def ver_fragmentos(nombre):
    ruta = os.path.join(BASE_DIR, "documents", "metadatos.pkl")
    fragmentos = []

    if os.path.exists(ruta):
        with open(ruta, "rb") as f:
            metadatos = pickle.load(f)
            for idx, doc in enumerate(metadatos):
                if doc.get("nombre") == nombre:
                    frag = {**doc, "id": idx}
                    fragmentos.append(frag)

    return render_template("fragmentos_documento.html", nombre=nombre, fragmentos=fragmentos)
