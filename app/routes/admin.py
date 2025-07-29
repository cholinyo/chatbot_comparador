from flask import Blueprint, render_template, request, redirect, flash, url_for, jsonify
import os
import json
import logging
import time
import subprocess
from datetime import datetime

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

def get_ollama_models():
    """Obtiene lista de modelos disponibles en Ollama"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        available_models = []
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    available_models.append(model_name)
        return available_models
    except Exception as e:
        logging.error(f"Error obteniendo modelos Ollama: {e}")
        return ["llama3.1:8b", "mistral:7b", "gemma:7b"]  # Fallback

@admin_bp.route("/admin", methods=["GET", "POST"])
def admin():
    config = cargar_config()
    modelos_disponibles = listar_modelos()

    if request.method == "POST":
        if "probar_openai" in request.form:
            # 🧪 Verificar acceso a OpenAI
            try:
                from app.services.bot_openai import OpenAIService
                service = OpenAIService()
                test_result = service.test_connection()
                
                if test_result["conectado"]:
                    flash(f"✅ OpenAI: {test_result['mensaje']}", "success")
                    logging.info("Prueba OpenAI exitosa")
                else:
                    flash(f"❌ OpenAI: {test_result['mensaje']}", "danger")
                    logging.error("Error en prueba OpenAI")
                    
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
        modelo_actual=config.get("modelo_local", ""),
        modelo_openai=config.get("modelo_openai", "gpt-4"),
        modelos_openai=MODELOS_OPENAI
    )

@admin_bp.route("/admin/model-tuning", methods=["GET", "POST"])
def model_tuning():
    """Configuración avanzada de parámetros de modelos"""
    
    if request.method == "POST":
        try:
            # Guardar configuración de modelos
            config = cargar_config()
            
            # Parámetros OpenAI
            config["openai_params"] = {
                "temperature": float(request.form.get("openai_temperature", 0.7)),
                "max_tokens": int(request.form.get("openai_max_tokens", 512)),
                "top_p": float(request.form.get("openai_top_p", 1.0)),
                "frequency_penalty": float(request.form.get("openai_frequency_penalty", 0.0)),
                "presence_penalty": float(request.form.get("openai_presence_penalty", 0.0))
            }
            
            # Parámetros modelo local
            config["local_params"] = {
                "temperature": float(request.form.get("local_temperature", 0.3)),
                "max_tokens": int(request.form.get("local_max_tokens", 512)),
                "top_k": int(request.form.get("local_top_k", 40)),
                "top_p": float(request.form.get("local_top_p", 0.7)),
                "n_threads": int(request.form.get("local_n_threads", 6)),
                "n_gpu_layers": int(request.form.get("local_n_gpu_layers", 0))
            }
            
            # Modelo activo
            active_model = request.form.get("active_model")
            if active_model:
                config["active_local_model"] = active_model
            
            guardar_config(config)
            flash("✅ Configuración de modelos actualizada", "success")
            logging.info("Configuración de model tuning actualizada")
            
            return redirect(url_for("admin.model_tuning"))
            
        except ValueError as e:
            flash(f"❌ Error en parámetros: {str(e)}", "danger")
        except Exception as e:
            flash(f"❌ Error al guardar configuración: {str(e)}", "danger")
            logging.error(f"Error en model_tuning: {e}")
    
    # Cargar configuración actual
    config = cargar_config()
    
    # Obtener modelos disponibles
    available_models = get_ollama_models()
    
    # Valores por defecto si no existen
    if "openai_params" not in config:
        config["openai_params"] = {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    
    if "local_params" not in config:
        config["local_params"] = {
            "temperature": 0.3,
            "max_tokens": 512,
            "top_k": 40,
            "top_p": 0.7,
            "n_threads": 6,
            "n_gpu_layers": 0
        }
    
    return render_template("model_tuning.html", 
                         config=config, 
                         available_models=available_models)

@admin_bp.route("/admin/test-models", methods=["POST"])
def test_models():
    """Probar modelos con diferentes configuraciones"""
    prompt_test = request.form.get("test_prompt", "¿Qué es la inteligencia artificial?")
    
    # Cargar configuración actual
    config = cargar_config()
    
    results = {
        "prompt": prompt_test,
        "timestamp": datetime.now().isoformat(),
        "openai": {"error": None, "response": "", "time": 0, "tokens": 0},
        "local": {"error": None, "response": "", "time": 0, "tokens": 0}
    }
    
    # Probar OpenAI
    try:
        from app.services.bot_openai import OpenAIService
        service = OpenAIService()
        start_time = time.time()
        
        result = service.get_openai_response(
            prompt_test,
            **config.get("openai_params", {})
        )
        
        results["openai"] = {
            "error": None,
            "response": result["respuesta"],
            "time": result["tiempo_respuesta"],
            "tokens": result["tokens_usados"]["total"],
            "model": result["modelo"]
        }
        
    except Exception as e:
        results["openai"]["error"] = str(e)
        logging.error(f"Error probando OpenAI: {e}")
    
    # Probar modelo local
    try:
        from app.services.bot_local import OllamaService
        service = OllamaService()
        
        result = service.get_local_response(
            prompt_test,
            model=config.get("active_local_model", "llama3.1:8b"),
            **config.get("local_params", {})
        )
        
        results["local"] = {
            "error": None,
            "response": result["respuesta"],
            "time": result["tiempo_respuesta"],
            "tokens": result["tokens_estimados"],
            "model": result["modelo"]
        }
        
    except Exception as e:
        results["local"]["error"] = str(e)
        logging.error(f"Error probando modelo local: {e}")
    
    # Guardar resultado del test
    save_test_result(results)
    
    return jsonify(results)

@admin_bp.route("/admin/test-history", methods=["GET"])
def test_history():
    """Mostrar historial de tests"""
    try:
        test_log_path = os.path.join("logs", "model_tests.json")
        if os.path.exists(test_log_path):
            with open(test_log_path, "r", encoding="utf-8") as f:
                tests = json.load(f)
            # Ordenar por timestamp descendente (más recientes primero)
            tests.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return render_template("test_history.html", tests=tests[:20])  # Últimos 20
        else:
            return render_template("test_history.html", tests=[])
    except Exception as e:
        flash(f"❌ Error cargando historial: {str(e)}", "danger")
        return render_template("test_history.html", tests=[])

@admin_bp.route("/admin/clear-tests", methods=["POST"])
def clear_tests():
    """Limpiar historial de tests"""
    try:
        test_log_path = os.path.join("logs", "model_tests.json")
        if os.path.exists(test_log_path):
            os.remove(test_log_path)
        flash("✅ Historial de tests limpiado", "success")
    except Exception as e:
        flash(f"❌ Error limpiando historial: {str(e)}", "danger")
    
    return redirect(url_for("admin.test_history"))

@admin_bp.route("/admin/system-info", methods=["GET"])
def system_info():
    """Información del sistema"""
    info = {
        "ollama_status": check_ollama_status(),
        "available_models": get_ollama_models(),
        "disk_usage": get_disk_usage(),
        "system_resources": get_system_resources()
    }
    
    return render_template("system_info.html", info=info)

def save_test_result(test_result):
    """Guardar resultado de test en archivo JSON"""
    test_log_path = os.path.join("logs", "model_tests.json")
    os.makedirs("logs", exist_ok=True)
    
    try:
        if os.path.exists(test_log_path):
            with open(test_log_path, "r", encoding="utf-8") as f:
                tests = json.load(f)
        else:
            tests = []
        
        tests.append(test_result)
        
        # Mantener solo los últimos 100 tests
        tests = tests[-100:]
        
        with open(test_log_path, "w", encoding="utf-8") as f:
            json.dump(tests, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logging.error(f"Error guardando test result: {e}")

def check_ollama_status():
    """Verificar si Ollama está funcionando"""
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def get_disk_usage():
    """Obtener uso de disco de la carpeta models"""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(MODELS_DIR):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return f"{total_size / (1024**3):.2f} GB"
    except:
        return "N/A"

def get_system_resources():
    """Obtener información básica del sistema"""
    try:
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    except ImportError:
        return {"cpu_percent": "N/A", "memory_percent": "N/A", "disk_percent": "N/A"}