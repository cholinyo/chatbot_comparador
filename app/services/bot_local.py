import os
import logging
from llama_cpp import Llama
import requests
import json
from app.config.settings import get_local_model_path

logger = logging.getLogger(__name__)

# Variables globales para los modelos cargados
_llm_file = None  # Para modelos .gguf locales
_ollama_available = None  # Para verificar disponibilidad de Ollama

def check_ollama_available():
    """Verifica si Ollama está disponible en el sistema"""
    global _ollama_available
    if _ollama_available is None:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            _ollama_available = response.status_code == 200
            if _ollama_available:
                logger.info("✅ Ollama disponible en localhost:11434")
            else:
                logger.warning("⚠️ Ollama no responde correctamente")
        except Exception as e:
            _ollama_available = False
            logger.warning(f"⚠️ Ollama no disponible: {e}")
    return _ollama_available

def get_ollama_models():
    """Obtiene la lista de modelos disponibles en Ollama"""
    if not check_ollama_available():
        return []
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            logger.info(f"📋 Modelos Ollama disponibles: {models}")
            return models
        else:
            logger.error(f"❌ Error obteniendo modelos Ollama: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"❌ Error conectando con Ollama: {e}")
        return []

def get_local_response_ollama(prompt, model_name="llama3.2"):
    """Genera respuesta usando Ollama"""
    if not check_ollama_available():
        raise Exception("Ollama no está disponible. Asegúrate de que esté ejecutándose.")
    
    try:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_k": 40,
                "top_p": 0.7
            }
        }
        
        logger.info(f"🔵 Enviando prompt a Ollama (modelo: {model_name})")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            respuesta = result.get("response", "").strip()
            logger.info(f"✅ Respuesta Ollama generada: {len(respuesta)} caracteres")
            return respuesta
        else:
            raise Exception(f"Error HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        logger.error(f"❌ Error con Ollama: {e}")
        raise

def get_local_response_file(prompt):
    """Genera respuesta usando modelo .gguf local"""
    global _llm_file
    
    if _llm_file is None:
        model_path = get_local_model_path()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"⚠️ Modelo local no encontrado en: {model_path}")
        
        logger.info(f"🔄 Cargando modelo local: {model_path}")
        _llm_file = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=6,
            n_gpu_layers=0,
            verbose=False
        )
        logger.info("✅ Modelo local cargado correctamente")

    system_prompt = (
        "Eres un asistente de IA local especializado en administración pública. "
        "Responde de forma precisa y profesional basándote únicamente en la información proporcionada."
    )
    
    prompt_formatted = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    
    logger.info("🔵 Generando respuesta con modelo local (.gguf)")
    try:
        output = _llm_file(
            prompt_formatted, 
            max_tokens=512, 
            temperature=0.3, 
            top_k=40, 
            top_p=0.7, 
            stop=["</s>"]
        )
        respuesta = output["choices"][0]["text"].strip()
        logger.info(f"✅ Respuesta local generada: {len(respuesta)} caracteres")
        return respuesta
    except Exception as e:
        logger.error(f"❌ Error generando respuesta local: {e}")
        raise

def get_local_response(prompt, model_type="auto", model_name="llama3.2"):
    """
    Función principal para obtener respuesta de modelos locales
    
    Args:
        prompt (str): El prompt a procesar
        model_type (str): "ollama", "file", o "auto"
        model_name (str): Nombre del modelo (solo para Ollama)
    
    Returns:
        str: Respuesta generada
    """
    logger.info(f"🚀 get_local_response llamada - Tipo: {model_type}, Modelo: {model_name}")
    
    if model_type == "auto":
        # Priorizar Ollama si está disponible
        if check_ollama_available():
            model_type = "ollama"
            logger.info("🔄 Auto-selección: usando Ollama")
        else:
            model_type = "file"
            logger.info("🔄 Auto-selección: usando modelo .gguf")
    
    if model_type == "ollama":
        return get_local_response_ollama(prompt, model_name)
    elif model_type == "file":
        return get_local_response_file(prompt)
    else:
        raise ValueError(f"Tipo de modelo no válido: {model_type}")

# Funciones de utilidad para obtener información
def get_available_local_models():
    """Retorna diccionario con modelos locales disponibles"""
    models = {
        "ollama": [],
        "files": []
    }
    
    # Verificar modelos Ollama
    if check_ollama_available():
        models["ollama"] = get_ollama_models()
    
    # Verificar modelos .gguf
    try:
        model_path = get_local_model_path()
        if os.path.exists(model_path):
            models["files"] = [os.path.basename(model_path)]
    except:
        pass
    
    return models

def get_model_status():
    """Retorna el estado de disponibilidad de los modelos locales"""
    status = {
        "ollama_available": check_ollama_available(),
        "ollama_models": get_ollama_models() if check_ollama_available() else [],
        "file_model_available": False,
        "file_model_path": None
    }
    
    try:
        model_path = get_local_model_path()
        status["file_model_available"] = os.path.exists(model_path)
        status["file_model_path"] = model_path
    except:
        pass
    
    return status