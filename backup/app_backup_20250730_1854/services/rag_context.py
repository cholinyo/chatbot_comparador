import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Paths por fuente
SOURCES = {
    "documentos": "vectorstore/documents",
    "web": "vectorstore/web",
    "apis": "vectorstore/apis"
}

# Función para cargar fragmentos + metadatos + FAISS
def cargar_fuente(path):
    try:
        index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "fragmentos.pkl"), "rb") as f:
            fragmentos = pickle.load(f)
        with open(os.path.join(path, "metadatos.pkl"), "rb") as f:
            metadatos = pickle.load(f)
        return index, metadatos, fragmentos
    except:
        return None, [], []

# Cargamos todas las fuentes
vectores = {}
for nombre, ruta in SOURCES.items():
    index, metadatos, fragmentos = cargar_fuente(ruta)
    vectores[nombre] = {
        "index": index,
        "metadatos": metadatos,
        "fragmentos": fragmentos
    }

def recuperar_contexto(pregunta, k=5, fuente="todas"):
    """
    fuente: 'documentos', 'web', 'apis' o 'todas'
    """
    emb = model.encode([pregunta])
    resultados = []

    fuentes_a_usar = [fuente] if fuente in SOURCES else SOURCES.keys()

    for f in fuentes_a_usar:
        index = vectores[f]["index"]
        fragmentos = vectores[f]["fragmentos"]
        metadatos = vectores[f]["metadatos"]

        if index is None:
            continue

        dist, ind = index.search(emb, k)
        for i, idx in enumerate(ind[0]):
            if idx < len(fragmentos):
                resultados.append({
                    "texto": fragmentos[idx],
                    "metadata": metadatos[idx],
                    "distancia": float(dist[0][i]),
                    "fuente": f
                })

    resultados.sort(key=lambda x: x["distancia"])
    return resultados[:k]
