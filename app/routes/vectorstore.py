import os
import subprocess
import pickle
import numpy as np
import faiss
from flask import Blueprint, render_template, request, redirect, url_for, flash
from datetime import datetime
from sklearn.metrics.pairwise import pairwise_distances

vectorstore_bp = Blueprint("vectorstore", __name__)

def cargar_embeddings(ruta):
    try:
        return np.load(ruta)
    except Exception:
        return None

def cargar_fragmentos(ruta):
    try:
        with open(ruta, "rb") as f:
            return pickle.load(f)
    except Exception:
        return []

def obtener_fecha_actualizacion(ruta_archivo):
    if os.path.exists(ruta_archivo):
        ts = os.path.getmtime(ruta_archivo)
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
    return "N/A"

def contar_fuentes(metadatos):
    if isinstance(metadatos, list):
        return len(set(meta["documento"] for meta in metadatos if "documento" in meta))
    return 0

def analizar(emb):
    if emb is None or len(emb) == 0:
        return {"similitud_media": "N/A", "histograma": []}
    dists = pairwise_distances(emb)
    tril_indices = np.tril_indices(len(emb), k=-1)
    valores = dists[tril_indices]
    histograma = np.histogram(valores, bins=10, range=(0, 2))
    media = np.mean(valores) if len(valores) > 0 else 0
    return {"similitud_media": round(media, 4), "histograma": histograma[0].tolist()}

@vectorstore_bp.route("/vectorstore")
def vista_vectorstore():
    fuentes = {
        "documents": "vectorstore/documents",
        "web": "vectorstore/web",
        "apis": "vectorstore/apis"
    }

    datos = {}

    for fuente, ruta in fuentes.items():
        emb_path = os.path.join(ruta, "embeddings.npy")
        frag_path = os.path.join(ruta, "fragmentos.pkl")
        meta_path = os.path.join(ruta, "metadatos.pkl")

        embeddings = cargar_embeddings(emb_path)
        fragmentos = cargar_fragmentos(frag_path)
        try:
            with open(meta_path, "rb") as f:
                metadatos = pickle.load(f)
        except Exception:
            metadatos = []

        datos[fuente] = {
            "fragmentos": len(fragmentos),
            "dimensiones": embeddings.shape[1] if embeddings is not None else "N/A",
            "actualizacion": obtener_fecha_actualizacion(emb_path),
            "fuentes": contar_fuentes(metadatos),
            "analisis": analizar(embeddings)
        }

    return render_template("vectorstore.html", datos=datos)

@vectorstore_bp.route("/vectorstore/reindex/<fuente>", methods=["POST"])
def reindex_fuente(fuente):
    scripts = {
        "documents": "reindex_documents.bat",
        "web": "reindex_web.bat",
        "apis": "reindex_apis.bat"
    }

    if fuente not in scripts:
        flash("Fuente no válida", "danger")
        return redirect(url_for("vectorstore.vista_vectorstore"))

    try:
        script_path = os.path.join(os.getcwd(), scripts[fuente])
        result = subprocess.run(script_path, shell=True, check=True)
        flash(f"{fuente.capitalize()} reindexado correctamente", "success")
    except subprocess.CalledProcessError as e:
        flash(f"Error al reindexar {fuente}: {str(e)}", "danger")
    except Exception as e:
        flash(f"Error inesperado: {str(e)}", "danger")

    return redirect(url_for("vectorstore.vista_vectorstore"))
