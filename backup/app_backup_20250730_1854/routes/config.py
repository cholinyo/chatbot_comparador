"""
Config routes - Actualizado para ChromaDB + LlamaIndex
Mantiene compatibilidad con interfaz existente
"""
from flask import Blueprint, render_template, request, redirect, flash, jsonify
import os
import json
import logging

# Imports actualizados
from app.utils.rag_utils_updated import (
    ingest_documents_with_llamaindex,
    obtener_estadisticas_vectorstore,
    buscar_por_tipo_documento,
    obtener_tipos_documento_disponibles,
    reindexar_fuente
)

config_bp = Blueprint("config", __name__)
CONFIG_PATH = os.path.join("app", "config", "settings.json")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cargar_config():
    """Cargar configuración desde JSON"""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error cargando config: {e}")
        return {}

def guardar_config(data):
    """Guardar configuración a JSON"""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"❌ Error guardando config: {e}")
        return False

@config_bp.route("/config", methods=["GET", "POST"])
def config():
    """Página principal de configuración - Actualizada"""
    config = cargar_config()
    carpetas = config.get("document_folders", [])
    urls = config.get("web_sources", [])
    apis = config.get("api_sources", [])
    bases_datos = config.get("db_sources", [])
    rag_k = config.get("rag_k", 3)

    if request.method == "POST":
        accion = request.form.get("accion")

        # Guardar valor de RAG K
        if accion == "guardar_k":
            nuevo_k = request.form.get("rag_k")
            try:
                nuevo_k = int(nuevo_k)
                if 1 <= nuevo_k <= 10:
                    config["rag_k"] = nuevo_k
                    if guardar_config(config):
                        flash("✅ Parámetro RAG (k) actualizado correctamente", "success")
                    else:
                        flash("❌ Error guardando configuración", "danger")
                else:
                    flash("❌ El valor debe estar entre 1 y 10", "danger")
            except ValueError:
                flash("❌ El valor debe ser un número entero", "danger")
            return redirect("/config")

        # NUEVA FUNCIONALIDAD: Reindexar con LlamaIndex
        if accion == "reindexar_documentos":
            try:
                total_docs = ingest_documents_with_llamaindex(carpetas)
                if total_docs > 0:
                    flash(f"✅ Reindexados {total_docs} fragmentos con LlamaIndex + ChromaDB", "success")
                else:
                    flash("⚠️ No se encontraron documentos para reindexar", "warning")
            except Exception as e:
                flash(f"❌ Error en reindexación: {str(e)}", "danger")
            return redirect("/config")

        # Gestión de carpetas (sin cambios)
        nueva = request.form.get("nueva_carpeta")
        eliminar = request.form.get("eliminar_carpeta")

        if nueva:
            if os.path.exists(nueva):
                if nueva not in carpetas:
                    carpetas.append(nueva)
                    config["document_folders"] = carpetas
                    if guardar_config(config):
                        flash("📁 Carpeta añadida correctamente", "success")
                    else:
                        flash("❌ Error guardando configuración", "danger")
                else:
                    flash("⚠️ La carpeta ya está en la lista", "warning")
            else:
                flash("❌ La ruta no existe", "danger")

        elif eliminar:
            if eliminar in carpetas:
                carpetas.remove(eliminar)
                config["document_folders"] = carpetas
                if guardar_config(config):
                    flash("🗑️ Carpeta eliminada correctamente", "success")
                else:
                    flash("❌ Error guardando configuración", "danger")

        # Gestión de URLs (sin cambios)
        if accion == "add_url":
            nueva_url = request.form.get("nueva_url")
            profundidad = request.form.get("profundidad_url")

            try:
                profundidad = int(profundidad)
                if nueva_url and profundidad > 0:
                    if not any(u["url"] == nueva_url for u in urls):
                        urls.append({"url": nueva_url, "depth": profundidad})
                        config["web_sources"] = urls
                        if guardar_config(config):
                            flash("🌐 URL añadida correctamente", "success")
                        else:
                            flash("❌ Error guardando configuración", "danger")
                    else:
                        flash("⚠️ La URL ya está registrada", "warning")
                else:
                    flash("❌ URL inválida o profundidad no válida", "danger")
            except ValueError:
                flash("❌ La profundidad debe ser un número", "danger")

        elif request.form.get("eliminar_url"):
            eliminar_url = request.form.get("eliminar_url")
            urls = [u for u in urls if u["url"] != eliminar_url]
            config["web_sources"] = urls
            if guardar_config(config):
                flash("🗑️ URL eliminada correctamente", "success")
            else:
                flash("❌ Error guardando configuración", "danger")

        # Gestión de APIs (sin cambios)
        if accion == "add_api":
            name = request.form.get("api_name")
            url = request.form.get("api_url")
            auth = request.form.get("api_auth")
            env_key = request.form.get("api_env_key")

            if url and name:
                if not any(api["url"] == url for api in apis):
                    api_entry = {
                        "name": name,
                        "url": url,
                        "auth": auth
                    }
                    if auth == "env" and env_key:
                        api_entry["env_key"] = env_key
                    apis.append(api_entry)
                    config["api_sources"] = apis
                    if guardar_config(config):
                        flash(f"🔌 API '{name}' añadida correctamente", "success")
                    else:
                        flash("❌ Error guardando configuración", "danger")
                else:
                    flash("⚠️ La URL ya está registrada", "warning")
            else:
                flash("❌ Nombre o URL de la API no válidos", "danger")

        elif request.form.get("eliminar_api"):
            eliminar_url = request.form.get("eliminar_api")
            apis = [a for a in apis if a["url"] != eliminar_url]
            config["api_sources"] = apis
            if guardar_config(config):
                flash("🗑️ API eliminada correctamente", "success")
            else:
                flash("❌ Error guardando configuración", "danger")

        # Gestión de bases de datos (sin cambios)
        if accion == "add_db":
            name = request.form.get("db_name")
            uri = request.form.get("db_uri")
            query = request.form.get("db_query")

            if name and uri and query:
                if not any(db["name"] == name for db in bases_datos):
                    bases_datos.append({
                        "name": name,
                        "uri": uri,
                        "query": query
                    })
                    config["db_sources"] = bases_datos
                    if guardar_config(config):
                        flash(f"🗃️ Base de datos '{name}' añadida correctamente", "success")
                    else:
                        flash("❌ Error guardando configuración", "danger")
                else:
                    flash("⚠️ Ya existe una base de datos con ese nombre", "warning")
            else:
                flash("❌ Todos los campos son obligatorios", "danger")

        elif request.form.get("eliminar_db"):
            eliminar_nombre = request.form.get("eliminar_db")
            bases_datos = [db for db in bases_datos if db["name"] != eliminar_nombre]
            config["db_sources"] = bases_datos
            if guardar_config(config):
                flash("🗑️ Base de datos eliminada correctamente", "success")
            else:
                flash("❌ Error guardando configuración", "danger")

        return redirect("/config")

    # NUEVA FUNCIONALIDAD: Obtener estadísticas de ChromaDB
    try:
        vectorstore_stats = obtener_estadisticas_vectorstore()
        tipos_documento = obtener_tipos_documento_disponibles()
    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas: {e}")
        vectorstore_stats = {"total_documents": 0, "error": str(e)}
        tipos_documento = []

    return render_template(
        "config.html", 
        carpetas=carpetas, 
        urls=urls, 
        apis=apis, 
        bases_datos=bases_datos, 
        rag_k=rag_k,
        vectorstore_stats=vectorstore_stats,
        tipos_documento=tipos_documento
    )

@config_bp.route("/ver_fragmentos", methods=["GET"])
def ver_fragmentos():
    """Ver fragmentos - Actualizado para ChromaDB"""
    origen = request.args.get("origen", "documentos")
    tipo_documento = request.args.get("tipo", None)
    
    origen_label = {
        "documentos": "📄 Ver fragmentos de documentos",
        "web": "🌐 Ver fragmentos de URLs",
        "apis": "🔌 Ver fragmentos de APIs",
        "bbdd": "🗃️ Ver fragmentos de bases de datos"
    }.get(origen, "Ver fragmentos")

    try:
        if tipo_documento:
            # Búsqueda por tipo específico
            from app.utils.rag_utils_updated import buscar_por_tipo_documento
            resultados = buscar_por_tipo_documento("", tipo_documento, k=100)
            fragmentos = [r["texto"] for r in resultados]
        else:
            # Búsqueda general por origen
            from app.utils.chroma_store import get_chroma_store
            store = get_chroma_store()
            
            # Filtrar por fuente si es necesario
            filtros = {"fuente": origen} if origen != "documentos" else {}
            sample_docs = store.search_by_metadata(filtros, limit=100)
            fragmentos = [doc.get("texto", "") for doc in sample_docs]

        fragmentos = fragmentos[:100]  # Límite de visualización
        
    except Exception as e:
        logger.error(f"❌ Error cargando fragmentos: {e}")
        flash(f"❌ Error cargando fragmentos: {str(e)}", "danger")
        fragmentos = []

    return render_template(
        "ver_fragmentos.html", 
        fragmentos=fragmentos, 
        origen=origen, 
        origen_label=origen_label,
        tipo_documento=tipo_documento
    )

@config_bp.route("/api/vectorstore/stats", methods=["GET"])
def api_vectorstore_stats():
    """API endpoint para estadísticas del vectorstore"""
    try:
        stats = obtener_estadisticas_vectorstore()
        tipos = obtener_tipos_documento_disponibles()
        
        return jsonify({
            "success": True,
            "stats": stats,
            "document_types": tipos
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@config_bp.route("/api/reindexar/<fuente>", methods=["POST"])
def api_reindexar_fuente(fuente):
    """API endpoint para reindexar fuentes específicas"""
    try:
        total = reindexar_fuente(fuente)
        
        return jsonify({
            "success": True,
            "message": f"Reindexados {total} documentos",
            "total_documents": total
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500