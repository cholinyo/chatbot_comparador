import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

VECTORSTORE_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")
META_PATH = os.path.join(VECTORSTORE_DIR, "metadatos.pkl")
TEXTOS_PATH = os.path.join(VECTORSTORE_DIR, "fragmentos.pkl")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Verificación de existencia
if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH) and os.path.exists(TEXTOS_PATH)):
    print("⚠️ No se encontró el índice vectorial. Ejecuta primero ingest_documents.py")
    metadatos = []
    fragmentos = []
    index = None
else:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadatos = pickle.load(f)
    with open(TEXTOS_PATH, "rb") as f:
        fragmentos = pickle.load(f)

def recuperar_contexto(pregunta, k=3):
    if index is None:
        return []

    emb = model.encode([pregunta])
    distancias, indices = index.search(emb, k)

    resultados = []
    for i, idx in enumerate(indices[0]):
        if idx < len(fragmentos):
            resultados.append({
                "texto": fragmentos[idx],
                "metadata": metadatos[idx],
                "distancia": float(distancias[0][i])
            })
    return resultados
