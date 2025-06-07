#Carga el índice FAISS (index.faiss) y los metadatos
#Utiliza sentence-transformers para convertir la pregunta en un embedding
#Devuelve los k fragmentos más cercanos semánticamente (por distancia L2)

import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Rutas
INDEX_PATH = os.path.join("vectorstore", "index.faiss")
META_PATH = os.path.join("vectorstore", "metadatos.pkl")

# Cargar modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Cargar índice y metadatos
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    metadatos = pickle.load(f)

def buscar_similares(pregunta, k=5):
    '''
    Dado un texto de entrada, retorna los k fragmentos más similares.
    '''
    emb = model.encode([pregunta])
    distancias, indices = index.search(emb, k)

    resultados = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadatos):
            resultados.append({
                "fragmento": metadatos[idx],
                "distancia": float(distancias[0][i])
            })
    return resultados

# Ejemplo de prueba
if __name__ == "__main__":
    consulta = input("🔍 Introduce tu pregunta: ")
    resultados = buscar_similares(consulta, k=3)
    for r in resultados:
        print(f"📄 Archivo: {r['fragmento']['archivo']} (fragmento #{r['fragmento']['fragmento']})")
        print(f"↔️ Distancia: {r['distancia']:.4f}")
        print("—" * 50)
