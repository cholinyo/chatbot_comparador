import os
import logging
import json
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from app.utils import doc_loader

CONFIG_PATH = os.path.join("app", "config", "settings.json")
VECTOR_DIR = os.path.join("vectorstore", "documents")
LOG_PATH = os.path.join("logs", "ingestion_documents.log")

os.makedirs("logs", exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)

def cargar_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    config = cargar_config()
    carpetas = config.get("document_folders", [])
    modelo_nombre = config.get("embedding_model", "all-MiniLM-L6-v2")

    if not carpetas:
        logging.warning("⚠️ No hay carpetas configuradas en settings.json")
        return

    modelo = SentenceTransformer(modelo_nombre)
    documentos = doc_loader.cargar_documentos(carpetas)

    if not documentos:
        logging.warning("⚠️ No se cargaron documentos desde las carpetas configuradas")
        return

    all_embeddings = []
    metadatos = []
    total_fragmentos = 0

    logging.info(f"📁 Iniciando ingesta de {len(documentos)} documentos")

    for doc in tqdm(documentos, desc="Procesando documentos"):
        bloques = doc.get("fragmentos", [])
        if not bloques:
            logging.warning(f"⚠️ Documento sin fragmentos: {doc['nombre']}")
            continue

        embeddings = modelo.encode(bloques, show_progress_bar=False)
        all_embeddings.extend(embeddings)

        metadatos.extend([{
            "nombre": doc["nombre"],
            "fragmento_id": i,
            "origen": "documento"
        } for i in range(len(bloques))])

        total_fragmentos += len(bloques)

    if not all_embeddings:
        logging.warning("❌ No se generaron embeddings")
        return

    embeddings_np = np.array(all_embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    np.save(os.path.join(VECTOR_DIR, "embeddings.npy"), embeddings_np)
    faiss.write_index(index, os.path.join(VECTOR_DIR, "index.faiss"))
    with open(os.path.join(VECTOR_DIR, "metadatos.pkl"), "wb") as f:
        pickle.dump(metadatos, f)

    logging.info(f"✅ Ingesta completada. Total fragmentos: {total_fragmentos}")

if __name__ == "__main__":
    main()