import os
import pickle
import faiss
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Configurar rutas
VECTOR_DIR = os.path.join("vectorstore", "web")
os.makedirs(VECTOR_DIR, exist_ok=True)

# Inicializar modelo y estructuras
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index_path = os.path.join(VECTOR_DIR, "index.faiss")
metadata_path = os.path.join(VECTOR_DIR, "fragmentos.pkl")

if os.path.exists(index_path) and os.path.exists(metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
else:
    index = faiss.IndexFlatL2(384)  # 384 dimensiones para MiniLM
    metadata = []

def save_index():
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

def partir_en_bloques(texto, max_caracteres=500):
    palabras = texto.split()
    fragmentos, fragmento = [], []
    for palabra in palabras:
        if sum(len(p) + 1 for p in fragmento) + len(palabra) < max_caracteres:
            fragmento.append(palabra)
        else:
            fragmentos.append(" ".join(fragmento))
            fragmento = [palabra]
    if fragmento:
        fragmentos.append(" ".join(fragmento))
    return fragmentos

def extraer_texto_web(url):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        driver.implicitly_wait(5)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        texto = soup.get_text(separator="\n")
        fragmentos = partir_en_bloques(texto)
        return list(set(fragmentos))
    finally:
        driver.quit()

def index_url(url, etiquetas="web"):
    try:
        fragmentos = extraer_texto_web(url)
        vectores = embedding_model.encode(fragmentos)
        index.add(vectores)
        metadata.extend([
            {"texto": frag, "fuente": "web", "url": url, "etiquetas": etiquetas} for frag in fragmentos
        ])
        save_index()
        return True, f"✅ Ingestada URL: {url}"
    except Exception as e:
        return False, f"❌ Error con {url}: {str(e)}"

if __name__ == "__main__":
    import json
    config_path = os.path.join("app", "config", "settings.json")

    if not os.path.exists(config_path):
        print("❌ No se encontró settings.json")
        exit()

    with open(config_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    fuentes = settings.get("web_sources", [])
    if not fuentes:
        print("⚠️ No hay URLs configuradas")
    else:
        for fuente in fuentes:
            url = fuente.get("url")
            print(f"🌐 Ingestando: {url}")
            exito, mensaje = index_url(url)
            print(mensaje)
