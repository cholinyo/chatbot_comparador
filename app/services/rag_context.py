import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Directorio de vectorstore para web y documentos
VECTOR_DIR_DOC = os.path.join("vectorstore", "documents")
VECTOR_DIR_WEB = os.path.join("vectorstore", "web")

model = SentenceTransformer("all-MiniLM-L6-v2")

def cargar_fragmentos_y_metadatos(directorio):
    index_path = os.path.join(directorio, "index.faiss")
    meta_path = os.path.join(directorio, "metadatos.pkl")
    textos_path = os.path.join(directorio, "fragmentos.pkl")

    if not (os.path.exists(index_path) and os.path.exists(meta_path) and os.path.exists(textos_path)):
        return None, [], [], f"⚠️ Índice no encontrado en {directorio}"

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadatos = pickle.load(f)
    with open(textos_path, "rb") as f:
        fragmentos = pickle.load(f)

    return index, metadatos, fragmentos, None

# Carga de índices
index_doc, metadatos_doc, fragmentos_doc, error_doc = cargar_fragmentos_y_metadatos(VECTOR_DIR_DOC)
index_web, metadatos_web, fragmentos_web, error_web = cargar_fragmentos_y_metadatos(VECTOR_DIR_WEB)

def recuperar_contexto(pregunta, k=3, fuente="ambas"):
    resultados = []
    emb = model.encode([pregunta])

    if fuente in ["documentos", "ambas"] and index_doc:
        dist_doc, ind_doc = index_doc.search(emb, k)
        for i, idx in enumerate(ind_doc[0]):
            if idx < len(fragmentos_doc):
                resultados.append({
                    "texto": fragmentos_doc[idx],
                    "metadata": metadatos_doc[idx],
                    "distancia": float(dist_doc[0][i]),
                    "fuente": "documento"
                })

    if fuente in ["web", "ambas"] and index_web:
        dist_web, ind_web = index_web.search(emb, k)
        for i, idx in enumerate(ind_web[0]):
            if idx < len(fragmentos_web):
                resultados.append({
                    "texto": fragmentos_web[idx],
                    "metadata": metadatos_web[idx],
                    "distancia": float(dist_web[0][i]),
                    "fuente": "web"
                })

    resultados.sort(key=lambda x: x["distancia"])
    return resultados[:k]
