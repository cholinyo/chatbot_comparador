from flask import Blueprint, render_template, request, redirect, flash
import os
import json
import logging

admin_bp = Blueprint("admin", __name__)

SETTINGS_PATH = os.path.join("app", "config", "settings.json")
MODELS_DIR = os.path.join("models")

# Configurar logger para auditoría de admin
LOG_PATH = os.path.join("logs", "admin.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)

# 🔹 Lista de modelos OpenAI permitidos para el desplegable
MODELOS_OPENAI = [
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4-0125-preview",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106"
]

# Carga la configuración actual
def cargar_config():
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# Guarda la nueva configuración
def guardar_config(data):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# Obtiene todos los .gguf disponibles
def listar_modelos():
    modelos = []
    for root, _, files in os.walk(MODELS_DIR):
        for file in files:
            if file.endswith(".gguf"):
                ruta = os.path.relpath(os.path.join(root, file), MODELS_DIR)
                modelos.append(ruta.replace("\\", "/"))  # normaliza para web
    return modelos

@admin_bp.route("/admin", methods=["GET", "POST"])
def admin():
    config = cargar_config()
    modelos_disponibles = listar_modelos()

    if request.method == "POST":
        if "probar_openai" in request.form:
            # 🧪 Verificar acceso a OpenAI
            import openai
            from dotenv import load_dotenv
            load_dotenv()

            openai.api_key = os.getenv("OPENAI_API_KEY")
            try:
                modelo = config.get("modelo_openai", "gpt-4")
                response = openai.ChatCompletion.create(
                    model=modelo,
                    messages=[{"role": "user", "content": "Ping"}],
                    max_tokens=5
                )
                flash(f"✅ OpenAI responde con modelo {modelo}: OK", "success")
                logging.info(f"Prueba OpenAI exitosa con modelo: {modelo}")

            except Exception as e:
                flash(f"❌ Error al conectar con OpenAI: {str(e)}", "danger")
                logging.error(f"❌ Error al probar conexión OpenAI: {str(e)}")

        else:
            # 🧩 Guardar configuración
            modelo_local = request.form.get("modelo_local")
            modelo_openai = request.form.get("modelo_openai")
            cambios = False

            if modelo_local and modelo_local in modelos_disponibles:
                config["modelo_local"] = modelo_local
                cambios = True
                logging.info(f"Modelo local actualizado a: {modelo_local}")

            if modelo_openai and modelo_openai in MODELOS_OPENAI:
                config["modelo_openai"] = modelo_openai
                cambios = True
                logging.info(f"Modelo OpenAI actualizado a: {modelo_openai}")

            if cambios:
                guardar_config(config)
                flash("✅ Configuración actualizada correctamente.", "success")
                return redirect("/admin")

    return render_template(
        "admin.html",
        modelos=modelos_disponibles,
        modelo_actual=config["modelo_local"],
        modelo_openai=config["modelo_openai"],
        modelos_openai=MODELOS_OPENAI
    )
