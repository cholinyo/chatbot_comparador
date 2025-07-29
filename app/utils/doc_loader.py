import os
import json
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.html import partition_html
from PyPDF2 import PdfReader

SETTINGS_PATH = os.path.join("app", "config", "settings.json")

# --- Funciones de lectura de archivos ---

def leer_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def leer_docx(path):
    elements = partition_docx(filename=path)
    return "\n".join([str(el) for el in elements])

def leer_pdf(path):
    try:
        elements = partition_pdf(filename=path)
        return "\n".join([str(el) for el in elements])
    except Exception as e:
        print(f"⚠️ unstructured falló en {path}, usando PyPDF2: {e}")
        try:
            reader = PdfReader(path)
            texto = ""
            for pagina in reader.pages:
                texto += pagina.extract_text() + "\n"
            return texto
        except Exception as e2:
            print(f"❌ PyPDF2 también falló en {path}: {e2}")
            return ""

def leer_html(path):
    elements = partition_html(filename=path)
    return "\n".join([str(el) for el in elements])

# --- Utilidad: partir texto en bloques ---

def partir_en_bloques(texto, max_caracteres=500):
    palabras = texto.split()
    bloques = []
    bloque_actual = ""

    for palabra in palabras:
        if len(bloque_actual) + len(palabra) + 1 <= max_caracteres:
            bloque_actual += " " + palabra
        else:
            bloques.append(bloque_actual.strip())
            bloque_actual = palabra
    if bloque_actual:
        bloques.append(bloque_actual.strip())

    return bloques

# --- Carga desde settings.json si se desea usar en modo autónomo ---

def cargar_config():
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Función principal ---

def cargar_documentos(carpetas):
    documentos = []
    for carpeta in carpetas:
        carpeta_path = Path(carpeta)
        if not carpeta_path.exists():
            print(f"⚠️ Carpeta no encontrada: {carpeta}")
            continue

        for ruta in carpeta_path.rglob("*"):
            if not ruta.is_file() or ruta.name.startswith("~$"):
                continue

            try:
                if ruta.suffix.lower() == ".txt":
                    texto = leer_txt(ruta)
                elif ruta.suffix.lower() == ".docx":
                    texto = leer_docx(ruta)
                elif ruta.suffix.lower() == ".pdf":
                    texto = leer_pdf(ruta)
                elif ruta.suffix.lower() == ".html":
                    texto = leer_html(ruta)
                else:
                    continue

                bloques = partir_en_bloques(texto)
                documentos.append({
                    "nombre": ruta.name,
                    "ruta": str(ruta),
                    "fragmentos": bloques
                })
            except Exception as e:
                print(f"❌ Error al procesar {ruta.name}: {e}")
                continue

    return documentos
