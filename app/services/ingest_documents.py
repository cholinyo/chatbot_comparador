import os
import logging
import json
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils import doc_loader

# Crear carpetas necesarias
VECTOR_DIR = os.path.join("vectorstore", "documents")
os.makedirs("logs", exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# Cargar configuración
CONFIG_PATH = os.path.join("app", "config", "settings.json")
def cargar_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
config = cargar_config()

# Configurar logging
LOG_PATH = os.path.join("logs", "ingestion_documents.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)

# Cargar modelo de embeddings desde config
modelo_embedding = config.get("embedding_model", "all-MiniLM-L6-v2")
modelo = SentenceTransformer(modelo_embedding)

# Crear splitter semántico
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ".", " "]
)

def limpiar_texto(texto):
    import re
    texto = re.sub(r"\s+", " ", texto)
    texto = texto.replace("\xa0", " ").strip()
    return texto

def main():
    carpetas = config.get("document_folders", [])
    if not carpetas:
        logging.warning("⚠️ No hay carpetas configuradas en settings.json")
        return

    documentos = doc_loader.cargar_documentos(carpetas)
    all_embeddings = []
    fragmento_metadatos = []
    total_fragmentos = 0

    logging.info(f"📁 Iniciando ingesta de {len(documentos)} documentos")

    for doc in tqdm(documentos, desc="Procesando documentos"):
        bloques_raw = doc.get("fragmentos", [])
        bloques_limpios = [limpiar_texto(b) for b in bloques_raw]
        bloques = []
        for b in bloques_limpios:
            bloques.extend(splitter.split_text(b))

        if not bloques:
            logging.warning(f"⚠️ Documento sin fragmentos: {doc['nombre']}")
            continue

        embeddings = modelo.encode(bloques, show_progress_bar=False)
        all_embeddings.extend(embeddings)

        for bloque in bloques:
            fragmento_metadatos.append({
                "documento": doc["nombre"],
                "fragmento": bloque,
                "origen": "documento",
                "etiquetas": ["documento"]
            })

        total_fragmentos += len(bloques)

    if not all_embeddings:
        logging.warning("❌ No se generaron embeddings")
        return

    embeddings_np = np.array(all_embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    np.save(os.path.join(VECTOR_DIR, "embeddings.npy"), embeddings_np)
    faiss.write_index(index, os.path.join(VECTOR_DIR, "index.faiss"))

    with open(os.path.join(VECTOR_DIR, "fragmentos.pkl"), "wb") as f:
        fragmentos = [m["fragmento"] for m in fragmento_metadatos]
        pickle.dump(fragmentos, f)

    with open(os.path.join(VECTOR_DIR, "metadatos.pkl"), "wb") as f:
        pickle.dump(fragmento_metadatos, f)

    logging.info(f"✅ Ingesta completada. Total fragmentos: {total_fragmentos}")

if __name__ == "__main__":
    main()