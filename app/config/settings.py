import os
import json
import logging

logger = logging.getLogger(__name__)
SETTINGS_PATH = os.path.join("app", "config", "settings.json")

def load_settings():
    """Carga la configuración desde settings.json"""
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"⚠️ Archivo {SETTINGS_PATH} no encontrado. Creando configuración por defecto.")
        return create_default_settings()
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error en JSON de configuración: {e}")
        return create_default_settings()

def save_settings(data):
    """Guarda la configuración en settings.json"""
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("✅ Configuración guardada correctamente")
    except Exception as e:
        logger.error(f"❌ Error guardando configuración: {e}")
        raise

def create_default_settings():
    """Crea una configuración por defecto"""
    default_config = {
        "modelo_local": "llama3-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "modelo_openai": "gpt-4",
        "default_model_type": "local",  # "local" o "openai"
        "rag_k": 5,
        "document_folders": [],
        "web_sources": [],
        "api_sources": [],
        "db_sources": [],
        "model_preferences": {
            "chat_default": "local",
            "comparador_local": "auto",
            "comparador_openai": "gpt-4"
        },
        "ollama_config": {
            "endpoint": "http://localhost:11434",
            "default_model": "llama3.2",
            "timeout": 60
        },
        "openai_config": {
            "default_model": "gpt-4",
            "max_tokens": 512,
            "temperature": 0.7
        }
    }
    
    # Guardar configuración por defecto
    save_settings(default_config)
    return default_config

def get_openai_model():
    """Obtiene el modelo OpenAI configurado"""
    config = load_settings()
    return config.get("modelo_openai", "gpt-4")

def get_local_model_path():
    """Obtiene la ruta completa del modelo local"""
    config = load_settings()
    modelo_relativo = config.get("modelo_local", "llama3-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    return os.path.join("models", modelo_relativo)

def get_default_model_type():
    """Obtiene el tipo de modelo por defecto"""
    config = load_settings()
    return config.get("default_model_type", "local")

def get_rag_k():
    """Obtiene el parámetro K para RAG"""
    config = load_settings()
    return config.get("rag_k", 5)

def get_model_preferences():
    """Obtiene las preferencias de modelos para diferentes funciones"""
    config = load_settings()
    return config.get("model_preferences", {
        "chat_default": "local",
        "comparador_local": "auto",
        "comparador_openai": "gpt-4"
    })

def get_ollama_config():
    """Obtiene la configuración de Ollama"""
    config = load_settings()
    return config.get("ollama_config", {
        "endpoint": "http://localhost:11434",
        "default_model": "llama3.2",
        "timeout": 60
    })

def get_openai_config():
    """Obtiene la configuración de OpenAI"""
    config = load_settings()
    return config.get("openai_config", {
        "default_model": "gpt-4",
        "max_tokens": 512,
        "temperature": 0.7
    })

def update_model_preference(function, model_type):
    """Actualiza la preferencia de modelo para una función específica"""
    config = load_settings()
    if "model_preferences" not in config:
        config["model_preferences"] = {}
    
    config["model_preferences"][function] = model_type
    save_settings(config)
    logger.info(f"✅ Preferencia de modelo actualizada: {function} -> {model_type}")

def is_openai_enabled():
    """Verifica si OpenAI está habilitado en la configuración"""
    api_key = os.getenv("OPENAI_API_KEY")
    return api_key is not None and api_key.startswith("sk-")

def get_available_models_config():
    """Obtiene la configuración de modelos disponibles"""
    config = load_settings()
    
    available = {
        "local": {
            "enabled": True,
            "ollama_endpoint": get_ollama_config()["endpoint"],
            "file_model": get_local_model_path()
        },
        "openai": {
            "enabled": is_openai_enabled(),
            "default_model": get_openai_model()
        }
    }
    
    return available

# Alias para compatibilidad retroactiva
cargar_config = load_settings
guardar_config = save_settings