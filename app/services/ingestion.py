import os
import faiss
import pickle
import requests

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf
from docx import Document as DocxDocument

DATA_DIR = "indexed_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Inicialización o carga del índice
index_path = os.path.join(DATA_DIR, "faiss.index")
meta_path = os.path.join(DATA_DIR, "metadata.pkl")

if os.path.exists(index_path) and os.path.exists(meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
else:
    index = faiss.IndexFlatL2(384)  # 384 = dimensiones de all-MiniLM-L6-v2
    metadata = []

def save_index():
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

def _add_to_index(text_chunks, etiquetas):
    vectors = model.encode(text_chunks)
    index.add(vectors)
    metadata.extend([{"texto": t, "etiquetas": etiquetas} for t in text_chunks])
    save_index()

def index_document(file, etiquetas=""):
    ext = file.filename.lower().split(".")[-1]
    etiquetas = etiquetas.split(",")

    try:
        if ext == "pdf":
            elements = partition_pdf(file=file.stream)
            text = "\n".join([e.text for e in elements if e.text])
        elif ext == "txt":
            text = file.read().decode("utf-8")
        elif ext == "docx":
            doc = DocxDocument(file)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            return f"Formato no soportado: {ext}"

        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        _add_to_index(chunks, etiquetas)
        return "Documento indexado correctamente"
    except Exception as e:
        return f"Error al indexar documento: {str(e)}"

def index_url(url, etiquetas=""):
    etiquetas = etiquetas.split(",")
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        text = soup.get_text()
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        _add_to_index(chunks, etiquetas)
        return "URL indexada correctamente"
    except Exception as e:
        return f"Error al indexar URL: {str(e)}"

def index_api(endpoint, etiquetas=""):
    etiquetas = etiquetas.split(",")
    try:
        res = requests.get(endpoint)
        if res.headers.get("Content-Type", "").startswith("application/json"):
            json_data = res.json()
            text = str(json_data)
        else:
            text = res.text
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        _add_to_index(chunks, etiquetas)
        return "API indexada correctamente"
    except Exception as e:
        return f"Error al indexar API: {str(e)}"
    
def buscar_contexto(query, top_k=5, filtro_etiquetas=None):
    """
    Devuelve los fragmentos más relevantes desde el índice vectorial.
    :param query: texto de consulta
    :param top_k: número de resultados
    :param filtro_etiquetas: lista de etiquetas opcional para filtrar
    :return: lista de fragmentos de texto relevantes
    """
    if not index.is_trained or index.ntotal == 0:
        return ["⚠️ No hay datos indexados"]

    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)

    resultados = []
    for idx in indices[0]:
        if idx < len(metadata):
            meta = metadata[idx]
            if filtro_etiquetas:
                etiquetas = [e.strip() for e in meta.get("etiquetas", [])]
                if not any(e in etiquetas for e in filtro_etiquetas):
                    continue
            resultados.append(meta["texto"])

    return resultados if resultados else ["⚠️ No se encontraron coincidencias con esas etiquetas"]
