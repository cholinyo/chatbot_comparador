from flask import Blueprint, render_template, request, redirect, flash
import os
import json
import pickle
import faiss

config_bp = Blueprint("config", __name__)
CONFIG_PATH = os.path.join("app", "config", "settings.json")

def cargar_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def guardar_config(data):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

@config_bp.route("/config", methods=["GET", "POST"])
def config():
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
                    guardar_config(config)
                    flash("✅ Parámetro RAG (k) actualizado correctamente", "success")
                else:
                    flash("❌ El valor debe estar entre 1 y 10", "danger")
            except ValueError:
                flash("❌ El valor debe ser un número entero", "danger")
            return redirect("/config")

        # Gestión de carpetas
        nueva = request.form.get("nueva_carpeta")
        eliminar = request.form.get("eliminar_carpeta")

        if nueva:
            if os.path.exists(nueva):
                if nueva not in carpetas:
                    carpetas.append(nueva)
                    config["document_folders"] = carpetas
                    guardar_config(config)
                    flash("📁 Carpeta añadida correctamente", "success")
                else:
                    flash("⚠️ La carpeta ya está en la lista", "warning")
            else:
                flash("❌ La ruta no existe", "danger")

        elif eliminar:
            if eliminar in carpetas:
                carpetas.remove(eliminar)
                config["document_folders"] = carpetas
                guardar_config(config)
                flash("🗑️ Carpeta eliminada correctamente", "success")

        # Gestión de URLs
        if accion == "add_url":
            nueva_url = request.form.get("nueva_url")
            profundidad = request.form.get("profundidad_url")

            try:
                profundidad = int(profundidad)
                if nueva_url and profundidad > 0:
                    if not any(u["url"] == nueva_url for u in urls):
                        urls.append({"url": nueva_url, "depth": profundidad})
                        config["web_sources"] = urls
                        guardar_config(config)
                        flash("🌐 URL añadida correctamente", "success")
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
            guardar_config(config)
            flash("🗑️ URL eliminada correctamente", "success")

        # Gestión de APIs
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
                    guardar_config(config)
                    flash(f"🔌 API '{name}' añadida correctamente", "success")
                else:
                    flash("⚠️ La URL ya está registrada", "warning")
            else:
                flash("❌ Nombre o URL de la API no válidos", "danger")

        elif request.form.get("eliminar_api"):
            eliminar_url = request.form.get("eliminar_api")
            apis = [a for a in apis if a["url"] != eliminar_url]
            config["api_sources"] = apis
            guardar_config(config)
            flash("🗑️ API eliminada correctamente", "success")

        # Gestión de bases de datos
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
                    guardar_config(config)
                    flash(f"🗃️ Base de datos '{name}' añadida correctamente", "success")
                else:
                    flash("⚠️ Ya existe una base de datos con ese nombre", "warning")
            else:
                flash("❌ Todos los campos son obligatorios", "danger")

        elif request.form.get("eliminar_db"):
            eliminar_nombre = request.form.get("eliminar_db")
            bases_datos = [db for db in bases_datos if db["name"] != eliminar_nombre]
            config["db_sources"] = bases_datos
            guardar_config(config)
            flash("🗑️ Base de datos eliminada correctamente", "success")

        return redirect("/config")

    return render_template("config.html", carpetas=carpetas, urls=urls, apis=apis, bases_datos=bases_datos, rag_k=rag_k)

@config_bp.route("/ver_fragmentos", methods=["GET"])
def ver_fragmentos():
    VECTOR_DIR = os.path.join("vectorstore", "documents")
    index_path = os.path.join(VECTOR_DIR, "index.faiss")
    fragmentos_path = os.path.join(VECTOR_DIR, "fragmentos.pkl")

    if not os.path.exists(index_path) or not os.path.exists(fragmentos_path):
        flash("❌ No se encontró el índice o los fragmentos", "danger")
        return redirect("/config")

    with open(fragmentos_path, "rb") as f:
        fragmentos = pickle.load(f)

    fragmentos = fragmentos[:100]
    return render_template("ver_fragmentos.html", fragmentos=fragmentos)