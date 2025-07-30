import os
import subprocess
import pickle
import numpy as np
import faiss
from flask import Blueprint, render_template, request, redirect, url_for, flash
from datetime import datetime
from sklearn.metrics.pairwise import pairwise_distances

vectorstore_bp = Blueprint("vectorstore", __name__)

def cargar_embeddings(ruta):
    try:
        return np.load(ruta)
    except Exception:
        return None

def cargar_fragmentos(ruta):
    try:
        with open(ruta, "rb") as f:
            return pickle.load(f)
    except Exception:
        return []

def obtener_fecha_actualizacion(ruta_archivo):
    if os.path.exists(ruta_archivo):
        ts = os.path.getmtime(ruta_archivo)
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
    return "N/A"

def contar_fuentes(metadatos):
    if isinstance(metadatos, list):
        return len(set(meta["documento"] for meta in metadatos if "documento" in meta))
    return 0

def analizar(emb):
    if emb is None or len(emb) == 0:
        return {"similitud_media": "N/A", "histograma": []}
    dists = pairwise_distances(emb)
    tril_indices = np.tril_indices(len(emb), k=-1)
    valores = dists[tril_indices]
    histograma = np.histogram(valores, bins=10, range=(0, 2))
    media = np.mean(valores) if len(valores) > 0 else 0
    return {"similitud_media": round(media, 4), "histograma": histograma[0].tolist()}

@vectorstore_bp.route("/vectorstore")
def vista_vectorstore():
    fuentes = {
        "documents": "vectorstore/documents",
        "web": "vectorstore/web",
        "apis": "vectorstore/apis"
    }

    datos = {}

    for fuente, ruta in fuentes.items():
        emb_path = os.path.join(ruta, "embeddings.npy")
        frag_path = os.path.join(ruta, "fragmentos.pkl")
        meta_path = os.path.join(ruta, "metadatos.pkl")

        embeddings = cargar_embeddings(emb_path)
        fragmentos = cargar_fragmentos(frag_path)
        try:
            with open(meta_path, "rb") as f:
                metadatos = pickle.load(f)
        except Exception:
            metadatos = []

        datos[fuente] = {
            "fragmentos": len(fragmentos),
            "dimensiones": embeddings.shape[1] if embeddings is not None else "N/A",
            "actualizacion": obtener_fecha_actualizacion(emb_path),
            "fuentes": contar_fuentes(metadatos),
            "analisis": analizar(embeddings)
        }

    return render_template("vectorstore.html", datos=datos)


# Versión corregida de la función reindex_fuente con mejor debugging:

# Reemplaza tu función reindex_fuente con esta versión de debugging:

@vectorstore_bp.route("/vectorstore/reindex/<fuente>", methods=["POST"])
def reindex_fuente(fuente):
    """Reindexar fuente específica con debugging completo"""
    import logging
    import sys
    import traceback
    
    # Configurar logging para que aparezca en consola
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print(f"\n{'='*50}")
    print(f"🚀 INICIANDO REINDEXACIÓN DE: {fuente}")
    print(f"{'='*50}")
    
    try:
        success = False
        
        if fuente == "documents":
            print("📄 Procesando documentos...")
            logger.info("📄 Importando módulo de documentos...")
            
            try:
                from app.services.ingest_documents_improved import main
                print("✅ Módulo importado correctamente")
                logger.info("📄 Ejecutando ingesta de documentos...")
                success = main()
                print(f"📄 Resultado documentos: {success}")
                
            except ImportError as e:
                print(f"❌ Error importando módulo de documentos: {e}")
                # Intentar con el módulo original
                try:
                    from app.services.ingest_documents import main
                    print("🔄 Usando módulo original...")
                    success = main()
                except Exception as e2:
                    print(f"❌ Error con módulo original: {e2}")
                    success = False
                    
        elif fuente == "web":
            print("🌐 Procesando web...")
            logger.info("🌐 Importando módulo web...")
            from app.services.ingest_web import main
            print("✅ Módulo web importado")
            logger.info("🌐 Ejecutando ingesta web...")
            success = main()
            print(f"🌐 Resultado web: {success}")
            
        elif fuente == "apis":
            print("🔌 Procesando APIs...")
            logger.info("🔌 Importando módulo APIs...")
            from app.services.ingest_api import main
            print("✅ Módulo APIs importado")
            logger.info("🔌 Ejecutando ingesta APIs...")
            success = main()
            print(f"🔌 Resultado APIs: {success}")
            
        else:
            print(f"❌ Fuente no válida: {fuente}")
            flash("Fuente no válida", "danger")
            return redirect(url_for("vectorstore.vista_vectorstore"))
        
        # Mostrar resultado final
        print(f"\n{'='*30}")
        if success:
            print(f"✅ {fuente.upper()} REINDEXADO CORRECTAMENTE")
            flash(f"✅ {fuente.capitalize()} reindexado correctamente", "success")
        else:
            print(f"❌ FALLÓ LA REINDEXACIÓN DE {fuente.upper()}")
            flash(f"❌ Error al reindexar {fuente}", "danger")
        print(f"{'='*30}\n")
            
    except Exception as e:
        error_msg = f"❌ ERROR CRÍTICO en {fuente}: {str(e)}"
        print(f"\n{error_msg}")
        print("📋 TRACEBACK COMPLETO:")
        traceback.print_exc()
        
        logger.error(error_msg)
        flash(f"❌ Error inesperado: {str(e)}", "danger")

    print(f"🔚 FINALIZANDO REINDEXACIÓN DE {fuente.upper()}")
    return redirect(url_for("vectorstore.vista_vectorstore"))