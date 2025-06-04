# app/services/ingest_web.py
import os
import json
import time
import pickle
import numpy as np
import faiss
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from tqdm import tqdm

# Config
CONFIG_PATH = os.path.join("app", "config", "settings.json")
VECTOR_DIR = os.path.join("vectorstore", "web")
os.makedirs(VECTOR_DIR, exist_ok=True)

# Cargar configuración
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)
web_sources = config.get("web_sources", [])

# Inicializar Selenium
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
driver = webdriver.Chrome(options=options)

# Modelo de embeddings
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Función de extracción de texto renderizado
visited = set()
def crawl_and_extract(base_url, max_pages=3):
    dominio = urlparse(base_url).netloc
    to_visit = [base_url]
    texts = []

    while to_visit and len(texts) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            driver.get(url)
            time.sleep(1)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            text = soup.get_text(" ", strip=True)
            if len(text) > 100:
                texts.append({"url": url, "texto": text})
            visited.add(url)
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"])
                if urlparse(link).netloc == dominio and link not in visited:
                    to_visit.append(link)
        except Exception:
            continue
    return texts

# Fragmentación
def dividir_en_fragmentos(contenido, max_tokens=500):
    palabras = contenido.split()
    for i in range(0, len(palabras), max_tokens):
        yield " ".join(palabras[i:i + max_tokens])

# Ingestar URLs
todos_fragmentos = []
todos_embeddings = []
todos_metadatos = []

print("🌐 Iniciando ingesta de URLs...")
for entrada in tqdm(web_sources):
    url = entrada.get("url")
    depth = entrada.get("depth", 1)
    paginas = crawl_and_extract(url, max_pages=depth)
    for pagina in paginas:
        for frag in dividir_en_fragmentos(pagina["texto"]):
            todos_fragmentos.append(frag)
            todos_metadatos.append({
                "url": pagina["url"],
                "origen": "web",
                "longitud": len(frag)
            })

if todos_fragmentos:
    print(f"✅ Generando embeddings de {len(todos_fragmentos)} fragmentos...")
    embeddings = modelo.encode(todos_fragmentos, show_progress_bar=True)
    embeddings_np = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    faiss.write_index(index, os.path.join(VECTOR_DIR, "index.faiss"))
    with open(os.path.join(VECTOR_DIR, "fragmentos.pkl"), "wb") as f:
        pickle.dump(todos_fragmentos, f)
    with open(os.path.join(VECTOR_DIR, "metadatos.pkl"), "wb") as f:
        pickle.dump(todos_metadatos, f)
    print("✅ Ingesta web completada con éxito.")
else:
    print("⚠️ No se encontraron fragmentos válidos para indexar.")

# Cerrar navegador
driver.quit()