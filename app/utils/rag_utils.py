import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Cargar modelo de embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Buscar fragmentos desde una sola fuente
def load_vector_store(path):
    index_path = os.path.join(path, "index.faiss")
    metadata_path = os.path.join(path, "fragmentos.pkl")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None, None

    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadatos = pickle.load(f)

    return index, metadatos

# Buscar fragmentos relevantes desde documentos solamente
def buscar_fragmentos(consulta, k=3):
    index, metadatos = load_vector_store(os.path.join("vectorstore", "documents"))
    if index is None:
        return []

    embedding = embedding_model.encode([consulta]).astype("float32")
    distancias, indices = index.search(embedding, k)

    resultados = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadatos):
            resultado = metadatos[idx]
            resultados.append({
                "texto": resultado.get("texto", ""),
                "distancia": float(distancias[0][i]),
                "fuente": "documentos"
            })
    return resultados

# Buscar fragmentos relevantes desde todas las fuentes combinadas
def buscar_fragmentos_combinados(consulta, k=5):
    embedding = embedding_model.encode([consulta]).astype("float32")
    rutas = [
        ("vectorstore/documents", "documentos"),
        ("vectorstore/web", "web"),
        ("vectorstore/apis", "apis"),
        ("vectorstore/bbdd", "bbdd")
    ]

    resultados_totales = []

    for ruta, fuente in rutas:
        index, metadatos = load_vector_store(ruta)
        if index is None or not metadatos:
            continue

        distancias, indices = index.search(embedding, k)
        for i, idx in enumerate(indices[0]):
            if idx < len(metadatos):
                raw = metadatos[idx]
                if isinstance(raw, dict):
                    texto = raw.get("texto", str(raw))
                else:
                    texto = str(raw)

                resultados_totales.append({
                    "texto": texto,
                    "fuente": fuente,
                    "distancia": float(distancias[0][i])
                })

    # Ordenar por distancia (menor es mejor)
    resultados_totales.sort(key=lambda x: x["distancia"])
    return resultados_totales[:k]