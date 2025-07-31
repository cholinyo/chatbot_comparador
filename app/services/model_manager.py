"""
Gestor central de modelos para el chatbot
Controla qué modelo usar según la configuración del usuario
"""

import logging
import time
from app.services.bot_local import get_local_response, get_available_local_models, get_model_status
from app.services.bot_openai import get_openai_response, is_openai_configured

logger = logging.getLogger(__name__)

class ModelManager:
    """Gestor central para todos los modelos de lenguaje"""
    
    def __init__(self):
        self.default_local_model_type = "auto"  # "ollama", "file", o "auto"
        self.default_local_model_name = "llama3.2"
        self.default_openai_model = "gpt-4"
    
    def get_response(self, prompt, model_type="local", use_rag=True, question=None, **kwargs):
        """
        Genera respuesta usando el modelo especificado CON RAG
        
        Args:
            prompt (str): Prompt del usuario (puede ser pregunta simple o prompt con contexto RAG)
            model_type (str): "local", "openai", o modelo específico
            use_rag (bool): Si usar RAG para enriquecer el prompt
            question (str): Pregunta original (para RAG)
            **kwargs: Parámetros adicionales
        
        Returns:
            dict: {
                "response": str,
                "model_used": str,
                "time_taken": float,
                "success": bool,
                "error": str,
                "rag_fragments": list,
                "rag_used": bool
            }
        """
        start_time = time.time()
        result = {
            "response": "",
            "model_used": "",
            "time_taken": 0,
            "success": False,
            "error": None,
            "rag_fragments": [],
            "rag_used": False
        }
        
        # Aplicar RAG si está habilitado y tenemos una pregunta
        final_prompt = prompt
        if use_rag and question:
            try:
                from app.utils.rag_utils import buscar_fragmentos_combinados
                from app.config.settings import get_rag_k
                
                k = kwargs.get('rag_k', get_rag_k())
                fragments = buscar_fragmentos_combinados(question, k=k)
                
                if fragments:
                    context = "\n".join([f"- {f['texto']}" for f in fragments])
                    final_prompt = f"""Contexto de la administración local:

{context}

Pregunta del usuario: {question}

Instrucciones: Responde de forma precisa y profesional basándote en el contexto proporcionado. Si la información no está completa en el contexto, indícalo claramente pero proporciona la mejor respuesta posible."""
                    
                    result["rag_fragments"] = fragments
                    result["rag_used"] = True
                    logger.info(f"🔍 RAG aplicado: {len(fragments)} fragmentos recuperados")
                else:
                    logger.warning("⚠️ RAG no encontró fragmentos relevantes")
                    
            except Exception as e:
                logger.error(f"❌ Error aplicando RAG: {e}")
                # Continuar sin RAG si hay error
        
        try:
            if model_type == "local" or model_type.startswith("ollama:") or model_type.startswith("file:"):
                result.update(self._get_local_response(final_prompt, model_type, **kwargs))
            
            elif model_type == "openai" or model_type.startswith("openai:"):
                result.update(self._get_openai_response(final_prompt, model_type, **kwargs))
            
            else:
                # Intentar interpretar como modelo específico
                if ":" in model_type:
                    provider, model_name = model_type.split(":", 1)
                    if provider == "ollama":
                        result.update(self._get_local_response(final_prompt, "ollama", model_name=model_name))
                    elif provider == "openai":
                        result.update(self._get_openai_response(final_prompt, "openai", model=model_name))
                    else:
                        raise ValueError(f"Proveedor desconocido: {provider}")
                else:
                    # Por defecto, usar local
                    result.update(self._get_local_response(final_prompt, "local", **kwargs))
        
        except Exception as e:
            result["error"] = str(e)
            result["response"] = f"Error: {str(e)}"
            logger.error(f"❌ Error en ModelManager: {e}")
        
        finally:
            result["time_taken"] = time.time() - start_time
        
        return result
    
    def _get_local_response(self, prompt, model_type, **kwargs):
        """Procesa respuesta con modelos locales"""
        
        # Determinar tipo de modelo local
        if model_type.startswith("ollama:"):
            actual_type = "ollama"
            model_name = model_type.split(":", 1)[1]
        elif model_type.startswith("file:"):
            actual_type = "file"
            model_name = kwargs.get("model_name", self.default_local_model_name)
        else:
            actual_type = self.default_local_model_type
            model_name = kwargs.get("model_name", self.default_local_model_name)
        
        logger.info(f"🔵 ModelManager: Usando modelo local - Tipo: {actual_type}, Modelo: {model_name}")
        
        try:
            response = get_local_response(
                prompt, 
                model_type=actual_type, 
                model_name=model_name
            )
            
            return {
                "response": response,
                "model_used": f"local:{actual_type}:{model_name}",
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Error modelo local: {e}")
            raise
    
    def _get_openai_response(self, prompt, model_type, **kwargs):
        """Procesa respuesta con OpenAI - CON CONTROL ESTRICTO"""
        
        if not is_openai_configured():
            raise Exception("OpenAI no está configurado correctamente")
        
        model = kwargs.get("model", self.default_openai_model)
        if model_type.startswith("openai:"):
            model = model_type.split(":", 1)[1]
        
        logger.info(f"🔵 ModelManager: Usando OpenAI - Modelo: {model}")
        
        # IMPORTANTE: Usar force=True para autorizar la llamada
        response = get_openai_response(prompt, model=model, force=True)
        
        return {
            "response": response,
            "model_used": f"openai:{model}",
            "success": True
        }
    
    def get_available_models(self):
        """Obtiene todos los modelos disponibles"""
        models = {
            "local": get_available_local_models(),
            "openai": {
                "available": is_openai_configured(),
                "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"] if is_openai_configured() else []
            }
        }
        return models
    
    def get_system_status(self):
        """Obtiene el estado del sistema de modelos"""
        local_status = get_model_status()
        
        status = {
            "local": local_status,
            "openai": {
                "configured": is_openai_configured(),
                "api_key_present": bool(os.getenv("OPENAI_API_KEY"))
            }
        }
        
        return status
    
    def compare_models(self, prompt, models_to_compare):
        """
        Compara respuestas de múltiples modelos
        
        Args:
            prompt (str): Prompt a enviar
            models_to_compare (list): Lista de modelos ["local", "openai:gpt-4", etc.]
        
        Returns:
            dict: Resultados de comparación
        """
        results = {}
        
        for model in models_to_compare:
            logger.info(f"🔀 Comparando modelo: {model}")
            results[model] = self.get_response(prompt, model_type=model)
        
        return results

# Instancia global del gestor
model_manager = ModelManager()