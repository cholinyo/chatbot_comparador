
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Ruta directa a documentos
VECTOR_DIR = os.path.join("vectorstore", "documents")

# Cargar modelo de embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Cargar FAISS y metadatos
def load_vector_store():
    index_path = os.path.join(VECTOR_DIR, "index.faiss")
    metadata_path = os.path.join(VECTOR_DIR, "metadatos.pkl")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("Índice FAISS o metadatos no encontrados en vectorstore/documents")

    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadatos = pickle.load(f)

    return index, metadatos

# Buscar fragmentos relevantes a partir de una consulta
def buscar_fragmentos(consulta, k=3):
    index, metadatos = load_vector_store()
    embedding = embedding_model.encode([consulta]).astype("float32")
    distancias, indices = index.search(embedding, k)

    resultados = []
    for idx in indices[0]:
        if idx < len(metadatos):
            resultados.append(metadatos[idx])
    return resultados
