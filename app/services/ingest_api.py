import os
import json
import pickle
import numpy as np
import faiss
import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Rutas del vectorstore para APIs
VECTOR_DIR = os.path.join("vectorstore", "apis")
os.makedirs(VECTOR_DIR, exist_ok=True)

INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
FRAGMENTOS_PATH = os.path.join(VECTOR_DIR, "fragmentos.pkl")
EMBEDDINGS_PATH = os.path.join(VECTOR_DIR, "embeddings.npy")
METADATOS_PATH = os.path.join(VECTOR_DIR, "metadatos.pkl")
CONFIG_PATH = os.path.join("app", "config", "settings.json")

def cargar_config():
    """Carga el archivo de configuración settings.json"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def obtener_fragmentos_api(api_config):
    """Realiza una petición a una API REST y extrae los fragmentos de texto"""
    url = api_config.get("url")
    headers = api_config.get("headers", {})
    auth = api_config.get("auth")
    campo_texto = api_config.get("campo_texto", "texto")
    etiquetas = api_config.get("etiquetas", [])

    if auth:
        headers["Authorization"] = auth

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        datos = response.json()
        fragmentos = []
        metadatos = []

        if isinstance(datos, dict):
            datos = [datos]

        for item in datos:
            if campo_texto in item and isinstance(item[campo_texto], str):
                texto = item[campo_texto].strip()
                if texto:
                    fragmentos.append(texto)
                    metadatos.append({
                        "texto": texto,
                        "fuente": "api",
                        "api": api_config.get("nombre", url),
                        "url": url,
                        "etiquetas": etiquetas,
                        "json_original": item
                    })
        return fragmentos, metadatos
    except Exception as e:
        print(f"❌ Error al acceder a {url}: {str(e)}")
        return [], []

def guardar_salida(index, fragmentos, vectores, metadatos):
    """Guarda los archivos de salida en el vectorstore de APIs"""
    faiss.write_index(index, INDEX_PATH)
    with open(FRAGMENTOS_PATH, "wb") as f:
        pickle.dump(fragmentos, f)
    with open(METADATOS_PATH, "wb") as f:
        pickle.dump(metadatos, f)
    np.save(EMBEDDINGS_PATH, vectores)

def main():
    config = cargar_config()
    apis = config.get("api_sources", [])
    if not apis:
        print("⚠️ No hay APIs configuradas en settings.json")
        return

    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    todos_fragmentos = []
    todos_vectores = []
    todos_metadatos = []

    for api in tqdm(apis, desc="Ingestando APIs"):
        fragmentos, metadatos = obtener_fragmentos_api(api)
        if not fragmentos:
            continue
        vectores = modelo.encode(fragmentos).astype("float32")
        todos_fragmentos.extend([m["texto"] for m in metadatos])
        todos_vectores.extend(vectores)
        todos_metadatos.extend(metadatos)

    if not todos_vectores:
        print("❌ No se generaron embeddings")
        return

    index = faiss.IndexFlatL2(384)
    index.add(np.array(todos_vectores).astype("float32"))
    guardar_salida(index, todos_fragmentos, np.array(todos_vectores), todos_metadatos)
    print(f"✅ Ingesta completada. {len(todos_fragmentos)} fragmentos procesados.")

if __name__ == "__main__":
    main()
