import os
import pickle
import faiss
import numpy as np
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, SoupStrainer
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from sentence_transformers import SentenceTransformer

# Configuración de rutas
VECTOR_DIR = os.path.join("vectorstore", "web")
os.makedirs(VECTOR_DIR, exist_ok=True)

index_path = os.path.join(VECTOR_DIR, "index.faiss")
fragmentos_path = os.path.join(VECTOR_DIR, "fragmentos.pkl")
metadatos_path = os.path.join(VECTOR_DIR, "metadatos.pkl")
embeddings_path = os.path.join(VECTOR_DIR, "embeddings.npy")

# Inicializar modelo
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Almacenamiento temporal
fragmentos_totales = []
metadatos_totales = []
vectores_totales = []

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

def obtener_urls_del_html(html, base_url):
    soup = BeautifulSoup(html, "html.parser", parse_only=SoupStrainer("a"))
    urls = set()
    for tag in soup:
        if tag.name == "a" and tag.get("href"):
            href = urljoin(base_url, tag["href"])
            if urlparse(href).netloc == urlparse(base_url).netloc:
                urls.add(href.split("#")[0])
    return urls

def extraer_y_indexar_url(url):
    print(f"🔎 Visitando: {url}")
    try:
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        driver.implicitly_wait(8)
        html = driver.page_source
        driver.quit()
    except Exception as e:
        print(f"❌ Error al acceder a {url}: {str(e)}")
        return set()

    texto = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    fragmentos = partir_en_bloques(texto)
    if not fragmentos:
        print("⚠️ Página vacía.")
        return set()

    vectores = embedding_model.encode(fragmentos).astype("float32")

    fragmentos_totales.extend(fragmentos)
    vectores_totales.extend(vectores)
    metadatos_totales.extend([
        {
            "texto": frag,
            "fuente": "web",
            "url": url,
            "etiquetas": ["web"]
        } for frag in fragmentos
    ])

    nuevas_urls = obtener_urls_del_html(html, url)
    print(f"🔗 {len(nuevas_urls)} nuevos enlaces encontrados.")
    return nuevas_urls

def crawl_dominio(base_url, max_paginas=10):
    visitadas = set()
    pendientes = [base_url]
    urls_en_cola = set(pendientes)

    while pendientes and len(visitadas) < max_paginas:
        url = pendientes.pop(0)
        urls_en_cola.discard(url)

        if url in visitadas:
            continue

        nuevas_urls = extraer_y_indexar_url(url)
        visitadas.add(url)

        for nueva in nuevas_urls:
            if nueva not in visitadas and nueva not in urls_en_cola:
                pendientes.append(nueva)
                urls_en_cola.add(nueva)

    print(f"✅ Crawling finalizado. Total páginas visitadas: {len(visitadas)}")

def guardar_vectorstore():
    index = faiss.IndexFlatL2(384)
    index.add(np.array(vectores_totales).astype("float32"))

    faiss.write_index(index, index_path)
    with open(fragmentos_path, "wb") as f:
        pickle.dump(fragmentos_totales, f)
    with open(metadatos_path, "wb") as f:
        pickle.dump(metadatos_totales, f)
    np.save(embeddings_path, np.array(vectores_totales))

    print(f"✅ Vectorstore guardado en {VECTOR_DIR}")

# Punto de entrada
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
            max_paginas = fuente.get("depth", 10)
            print(f"\n🌐 Iniciando crawl para: {url} con depth={max_paginas}")
            crawl_dominio(url, max_paginas)

        guardar_vectorstore()
