import os
import logging
from flask import Blueprint, render_template, request, redirect, flash
from app.config.settings import load_settings, save_settings, get_available_models_config
from app.services.model_manager import model_manager
from app.services.bot_openai import test_openai_connection

logger = logging.getLogger(__name__)
admin_bp = Blueprint("admin", __name__)

def get_local_model_files():
    """Obtiene lista de archivos .gguf disponibles"""
    models_dir = "models"
    model_files = []
    
    if os.path.exists(models_dir):
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith(".gguf"):
                    relative_path = os.path.relpath(os.path.join(root, file), models_dir)
                    model_files.append(relative_path.replace("\\", "/"))
    
    return model_files

@admin_bp.route("/admin", methods=["GET", "POST"])
def admin():
    config = load_settings()
    
    if request.method == "POST":
        action = request.form.get("action")
        
        if action == "test_openai":
            # Probar conexión OpenAI
            success, message = test_openai_connection()
            if success:
                flash(f"✅ OpenAI: {message}", "success")
            else:
                flash(f"❌ OpenAI: {message}", "danger")
        
        elif action == "test_models":
            # Probar todos los modelos disponibles
            test_prompt = "¿Cuál es la capital de España?"
            
            # Probar modelos locales
            try:
                result = model_manager.get_response(test_prompt, model_type="local")
                if result["success"]:
                    flash(f"✅ Modelo local funciona: {result['model_used']}", "success")
                else:
                    flash(f"❌ Error modelo local: {result['error']}", "danger")
            except Exception as e:
                flash(f"❌ Error probando modelo local: {str(e)}", "danger")
            
            # Probar OpenAI si está configurado
            if config.get("test_openai_enabled", False):
                try:
                    result = model_manager.get_response(test_prompt, model_type="openai")
                    if result["success"]:
                        flash(f"✅ OpenAI funciona: {result['model_used']}", "success")
                    else:
                        flash(f"❌ Error OpenAI: {result['error']}", "danger")
                except Exception as e:
                    flash(f"❌ Error probando OpenAI: {str(e)}", "danger")
        
        elif action == "save_config":
            # Guardar configuración
            try:
                # Modelos
                if request.form.get("modelo_local"):
                    config["modelo_local"] = request.form.get("modelo_local")
                
                if request.form.get("modelo_openai"):
                    config["modelo_openai"] = request.form.get("modelo_openai")
                
                # Tipo de modelo por defecto
                if request.form.get("default_model_type"):
                    config["default_model_type"] = request.form.get("default_model_type")
                
                # Configuración Ollama
                ollama_endpoint = request.form.get("ollama_endpoint")
                ollama_model = request.form.get("ollama_default_model")
                
                if ollama_endpoint or ollama_model:
                    if "ollama_config" not in config:
                        config["ollama_config"] = {}
                    
                    if ollama_endpoint:
                        config["ollama_config"]["endpoint"] = ollama_endpoint
                    if ollama_model:
                        config["ollama_config"]["default_model"] = ollama_model
                
                # Configuración OpenAI
                openai_model = request.form.get("openai_default_model")
                openai_temp = request.form.get("openai_temperature")
                
                if openai_model or openai_temp:
                    if "openai_config" not in config:
                        config["openai_config"] = {}
                    
                    if openai_model:
                        config["openai_config"]["default_model"] = openai_model
                    if openai_temp:
                        config["openai_config"]["temperature"] = float(openai_temp)
                
                # RAG K
                rag_k = request.form.get("rag_k")
                if rag_k:
                    config["rag_k"] = int(rag_k)
                
                # Habilitar/deshabilitar pruebas OpenAI
                config["test_openai_enabled"] = bool(request.form.get("test_openai_enabled"))
                
                save_settings(config)
                flash("✅ Configuración guardada correctamente", "success")
                
            except Exception as e:
                flash(f"❌ Error guardando configuración: {str(e)}", "danger")
                logger.error(f"Error guardando configuración: {e}")
        
        return redirect("/admin")
    
    # Obtener información del sistema
    system_status = model_manager.get_system_status()
    available_models = model_manager.get_available_models()
    local_files = get_local_model_files()
    models_config = get_available_models_config()
    
    return render_template("admin.html",
                         config=config,
                         system_status=system_status,
                         available_models=available_models,
                         local_files=local_files,
                         models_config=models_config)

@admin_bp.route("/admin/model-test/<model_type>")
def test_specific_model(model_type):
    """Endpoint para probar un modelo específico"""
    test_prompt = "¿Cuál es la capital de Francia?"
    
    try:
        result = model_manager.get_response(test_prompt, model_type=model_type)
        
        return {
            "success": result["success"],
            "model": result["model_used"],
            "response": result["response"][:100] + "..." if len(result["response"]) > 100 else result["response"],
            "time": result["time_taken"],
            "error": result.get("error")
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@admin_bp.route("/admin/status")
def admin_status():
    """Endpoint para obtener estado actual del sistema"""
    return {
        "system_status": model_manager.get_system_status(),
        "available_models": model_manager.get_available_models(),
        "config": get_available_models_config()
    }