import os
import json
import logging
import pickle
import ssl
import numpy as np
import faiss
from tqdm import tqdm
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer
from app.utils import ingestion

# Configuración de entorno
ssl._create_default_https_context = ssl._create_unverified_context
CONFIG_PATH = os.path.join("app", "config", "settings.json")
VECTOR_DIR = os.path.join("vectorstore", "web")
LOG_PATH = os.path.join("logs", "ingestion_web.log")

# Crear carpetas necesarias
os.makedirs("logs", exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# Configurar logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)

def cargar_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def crawl_por_paginas(dominio, max_paginas):
    visitadas = set()
    por_visitar = [dominio]
    paginas = []

    while por_visitar and len(paginas) < max_paginas:
        url = por_visitar.pop(0)
        if url in visitadas:
            continue

        logging.info(f"🌐 Visitando: {url}")
        try:
            fragmentos = ingestion.extraer_texto(url)
            if fragmentos:
                paginas.append({"url": url, "fragmentos": fragmentos})
                logging.info(f"✅ {url} -> {len(fragmentos)} fragmentos")
        except Exception as e:
            logging.warning(f"❌ Error en {url}: {e}")

        try:
            enlaces = ingestion.obtener_enlaces(url)
            for enlace in enlaces:
                absoluto = urljoin(url, enlace)
                if urlparse(absoluto).netloc == urlparse(dominio).netloc and absoluto not in visitadas:
                    por_visitar.append(absoluto)
        except Exception as e:
            logging.warning(f"⚠️ Enlaces no extraídos de {url}: {e}")

        visitadas.add(url)

    logging.info(f"🔎 Total páginas visitadas desde {dominio}: {len(paginas)}")
    return paginas

def main():
    config = cargar_config()
    fuentes = config.get("web_sources", [])

    if not fuentes:
        logging.warning("⚠️ No hay URLs configuradas en settings.json")
        return

    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    all_embeddings = []
    metadatos = []
    total_fragmentos = 0

    for fuente in fuentes:
        url = fuente.get("url")
        k = int(fuente.get("depth", 1))
        if not url:
            continue

        logging.info(f"🚀 Procesando URL: {url} con máximo de páginas {k}")
        paginas = crawl_por_paginas(url, max_paginas=k)

        for pagina in tqdm(paginas, desc="Procesando fragmentos"):
            embeddings = modelo.encode(pagina["fragmentos"], show_progress_bar=False)
            all_embeddings.extend(embeddings)

            metadatos.append({
                "nombre": pagina["url"],
                "total_fragmentos": len(pagina["fragmentos"]),
                "origen": "url"
            })
            total_fragmentos += len(pagina["fragmentos"])

    if not all_embeddings:
        logging.warning("❌ No se generaron embeddings")
        return

    embeddings_np = np.array(all_embeddings).astype("float32")

    if embeddings_np.ndim != 2:
        logging.error(f"❌ Error en embeddings: forma inesperada {embeddings_np.shape}")
        return

    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    np.save(os.path.join(VECTOR_DIR, "embeddings.npy"), embeddings_np)
    faiss.write_index(index, os.path.join(VECTOR_DIR, "index.faiss"))
    with open(os.path.join(VECTOR_DIR, "metadatos.pkl"), "wb") as f:
        pickle.dump(metadatos, f)

    logging.info(f"🌟 Ingesta web completada. Total fragmentos: {total_fragmentos}")

if __name__ == "__main__":
    main()
