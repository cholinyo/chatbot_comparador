import os
import json
import logging
from bs4 import BeautifulSoup
import docx
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text

CONFIG_PATH = os.path.join("app", "config", "settings.json")
LOG_PATH = os.path.join("logs", "loader.log")
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)

def cargar_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def leer_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def leer_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def leer_pdf(path):
    try:
        elementos = partition_pdf(filename=path)
        return "\n".join(str(e) for e in elementos)
    except Exception as e:
        logging.warning(f"PDF no procesado: {path} - {e}")
        return ""

def leer_html(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text(separator="\n")
    except Exception as e:
        logging.warning(f"HTML no procesado: {path} - {e}")
        return ""

def partir_en_bloques(texto, max_caracteres=500):
    palabras = texto.split()
    fragmentos = []
    fragmento = []

    for palabra in palabras:
        if sum(len(p) + 1 for p in fragmento) + len(palabra) < max_caracteres:
            fragmento.append(palabra)
        else:
            fragmentos.append(" ".join(fragmento))
            fragmento = [palabra]
    if fragmento:
        fragmentos.append(" ".join(fragmento))

    return fragmentos

def cargar_documentos(rutas_directas=None):
    config = cargar_config()
    carpetas = rutas_directas or config.get("document_folders", [])
    documentos = []
    extensiones_validas = [".txt", ".pdf", ".docx", ".html"]

    for carpeta in carpetas:
        for root, _, files in os.walk(carpeta):
            for nombre in files:
                ext = os.path.splitext(nombre)[1].lower()
                if ext not in extensiones_validas:
                    continue

                ruta = os.path.join(root, nombre)
                texto = ""

                if ext == ".txt":
                    texto = leer_txt(ruta)
                elif ext == ".docx":
                    texto = leer_docx(ruta)
                elif ext == ".pdf":
                    texto = leer_pdf(ruta)
                elif ext == ".html":
                    texto = leer_html(ruta)

                if texto.strip():
                    bloques = partir_en_bloques(texto)
                    documentos.append({
                        "nombre": nombre,
                        "texto": "\n".join(bloques),
                        "origen": "documento"
                    })
                    logging.info(f"✅ {nombre} procesado ({len(bloques)} fragmentos)")
                else:
                    logging.warning(f"⚠️ {nombre} sin contenido procesable")

    logging.info(f"📄 Total documentos procesados: {len(documentos)}")
    return documentos
