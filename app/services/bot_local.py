"""
bot_local.py - Servicio mejorado para modelos LLM locales con Ollama
Parte del TFM: Prototipo de Chatbot Interno para Administraciones Locales
"""

import os
import time
import json
import logging
import subprocess
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaService:
    """Servicio para interactuar con modelos LLM locales usando Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.default_model = "llama3.1:8b"
        self.available_models = self._get_available_models()
        
        # Verificar que Ollama esté funcionando
        if not self._check_ollama_status():
            logger.warning("⚠️ Ollama no está disponible. Algunas funciones pueden fallar.")
    
    def _check_ollama_status(self) -> bool:
        """Verifica si Ollama está ejecutándose"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def _get_available_models(self) -> List[str]:
        """Obtiene lista de modelos disponibles en Ollama"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [model.get("name", "").split(":")[0] for model in data.get("models", [])]
                return list(set(models))  # Eliminar duplicados
            return []
        except Exception as e:
            logger.error(f"Error obteniendo modelos: {e}")
            return ["llama3.1", "mistral", "phi3"]  # Fallback
    
    def get_local_response(self,
                          prompt_usuario: str,
                          context_fragments: List[Dict] = None,
                          model: str = None,
                          temperature: float = 0.3,
                          max_tokens: int = 512,
                          top_k: int = 40,
                          top_p: float = 0.7,
                          **kwargs) -> Dict[str, Any]:
        """
        Genera respuesta usando modelo local con contexto RAG y métricas
        
        Args:
            prompt_usuario: Pregunta del usuario
            context_fragments: Lista de fragmentos RAG recuperados
            model: Modelo a usar
            temperature: Creatividad (0-1)
            max_tokens: Máximo tokens de respuesta
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            Dict con respuesta, métricas y metadatos
        """
        start_time = time.time()
        model_to_use = model or self.default_model
        
        try:
            # Construir prompt completo
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(prompt_usuario, context_fragments)
            full_prompt = self._format_prompt(system_prompt, user_prompt, model_to_use)
            
            # Configurar parámetros para Ollama
            payload = {
                "model": model_to_use,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                    "stop": ["</s>", "<|end|>", "Human:", "Usuario:"]
                }
            }
            
            # Realizar llamada a Ollama
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=120  # Timeout mayor para modelos locales
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                respuesta_texto = data.get("response", "").strip()
                
                # Limpiar respuesta
                respuesta_texto = self._clean_response(respuesta_texto)
                
                resultado = {
                    "respuesta": respuesta_texto,
                    "modelo": model_to_use,
                    "tiempo_respuesta": round(end_time - start_time, 2),
                    "tokens_estimados": self._estimate_tokens(respuesta_texto),
                    "parametros": {
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "max_tokens": max_tokens
                    },
                    "fragmentos_utilizados": len(context_fragments) if context_fragments else 0,
                    "exito": True,
                    "error": None,
                    "metadata": {
                        "eval_count": data.get("eval_count", 0),
                        "eval_duration": data.get("eval_duration", 0),
                        "load_duration": data.get("load_duration", 0)
                    }
                }
                
                logger.info(f"✅ Respuesta local generada - Modelo: {model_to_use}, Tiempo: {resultado['tiempo_respuesta']}s")
                return resultado
                
            else:
                raise Exception(f"Error HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            error_msg = f"❌ Error modelo local: {str(e)}"
            logger.error(error_msg)
            
            return {
                "respuesta": f"⚠️ Error en modelo local: {str(e)}",
                "modelo": model_to_use,
                "tiempo_respuesta": round(time.time() - start_time, 2),
                "tokens_estimados": 0,
                "parametros": {"temperature": temperature, "top_k": top_k, "top_p": top_p},
                "fragmentos_utilizados": 0,
                "exito": False,
                "error": str(e)
            }
    
    def _build_system_prompt(self) -> str:
        """Construye prompt del sistema para administración local"""
        return """Eres un asistente especializado en administración local española. 

INSTRUCCIONES IMPORTANTES:
- Responde ÚNICAMENTE basándote en la información proporcionada
- Sé preciso, claro y conciso
- Si no tienes información suficiente, dilo claramente
- No inventes datos o normativas
- Menciona las fuentes cuando sea relevante

FORMATO:
- Respuesta directa y práctica
- Máximo 3-4 párrafos
- Si hay dudas, recomienda verificación oficial

LIMITACIONES:
- No hagas interpretaciones legales definitivas
- No proporciones información que no esté en el contexto
- Sugiere contacto con departamentos específicos cuando sea necesario"""

    def _build_user_prompt(self, pregunta: str, fragmentos: List[Dict] = None) -> str:
        """Construye prompt del usuario con contexto RAG"""
        if not fragmentos:
            return f"Pregunta: {pregunta}\n\nResponde basándote en conocimiento general sobre administración local española."
        
        # Construir contexto
        contexto_partes = []
        for i, frag in enumerate(fragmentos[:5]):  # Máximo 5 fragmentos
            fuente = frag.get('fuente', 'desconocida')
            texto = frag.get('texto', '')
            contexto_partes.append(f"[Fuente {i+1} - {fuente}]: {texto}")
        
        contexto = "\n\n".join(contexto_partes)
        
        return f"""INFORMACIÓN DISPONIBLE:
{contexto}

PREGUNTA DEL USUARIO:
{pregunta}

INSTRUCCIONES:
Responde basándote únicamente en la información proporcionada. Si la información es insuficiente, indícalo claramente."""

    def _format_prompt(self, system: str, user: str, model: str) -> str:
        """Formatea el prompt según el modelo"""
        if "llama" in model.lower():
            return f"<|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n"
        elif "mistral" in model.lower():
            return f"<s>[INST] {system}\n\n{user} [/INST]"
        elif "phi" in model.lower():
            return f"System: {system}\n\nUser: {user}\n\nAssistant:"
        else:
            # Formato genérico
            return f"System: {system}\n\nUser: {user}\n\nAssistant:"
    
    def _clean_response(self, respuesta: str) -> str:
        """Limpia la respuesta del modelo"""
        # Eliminar tokens de parada comunes
        stop_tokens = ["</s>", "<|end|>", "Human:", "Usuario:", "System:", "Assistant:"]
        
        for token in stop_tokens:
            if token in respuesta:
                respuesta = respuesta.split(token)[0]
        
        # Limpiar espacios y líneas vacías
        lineas = [linea.strip() for linea in respuesta.split('\n')]
        lineas = [linea for linea in lineas if linea]
        
        return '\n'.join(lineas).strip()
    
    def _estimate_tokens(self, texto: str) -> int:
        """Estima número de tokens (aproximación)"""
        # Aproximación: 1 token ≈ 4 caracteres en español
        return len(texto) // 4
    
    def test_model(self, model: str = None) -> Dict[str, Any]:
        """Prueba un modelo específico"""
        model_to_test = model or self.default_model
        
        try:
            test_prompt = "¿Cuál es la capital de España? Responde brevemente."
            resultado = self.get_local_response(test_prompt, model=model_to_test)
            
            return {
                "modelo": model_to_test,
                "disponible": resultado["exito"],
                "tiempo_respuesta": resultado["tiempo_respuesta"],
                "mensaje": "✅ Modelo funcionando correctamente" if resultado["exito"] else "❌ Error en el modelo"
            }
            
        except Exception as e:
            return {
                "modelo": model_to_test,
                "disponible": False,
                "tiempo_respuesta": 0,
                "mensaje": f"❌ Error: {str(e)}"
            }
    
    def pull_model(self, model_name: str) -> bool:
        """Descarga un modelo si no está disponible"""
        try:
            logger.info(f"📥 Descargando modelo {model_name}...")
            
            payload = {"name": model_name}
            response = requests.post(f"{self.api_url}/pull", json=payload, stream=True)
            
            if response.status_code == 200:
                logger.info(f"✅ Modelo {model_name} descargado correctamente")
                self.available_models = self._get_available_models()  # Actualizar lista
                return True
            else:
                logger.error(f"❌ Error descargando {model_name}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error en pull_model: {e}")
            return False
    
    def get_model_info(self, model: str = None) -> Dict[str, Any]:
        """Obtiene información detallada de un modelo"""
        model_name = model or self.default_model
        
        try:
            payload = {"name": model_name}
            response = requests.post(f"{self.api_url}/show", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Modelo {model_name} no encontrado"}
                
        except Exception as e:
            return {"error": str(e)}


# Funciones de compatibilidad con el código existente
def get_local_response(prompt_usuario: str, **kwargs) -> str:
    """Función de compatibilidad que devuelve solo el texto"""
    try:
        service = OllamaService()
        resultado = service.get_local_response(prompt_usuario, **kwargs)
        return resultado["respuesta"]
    except Exception as e:
        logger.error(f"Error en get_local_response: {e}")
        return f"⚠️ Error modelo local: {str(e)}"


def get_detailed_local_response(prompt_usuario: str,
                               context_fragments: List[Dict] = None,
                               **kwargs) -> Dict[str, Any]:
    """Función para obtener respuesta detallada con métricas"""
    try:
        service = OllamaService()
        return service.get_local_response(prompt_usuario, context_fragments, **kwargs)
    except Exception as e:
        logger.error(f"Error en get_detailed_local_response: {e}")
        return {
            "respuesta": f"⚠️ Error: {str(e)}",
            "exito": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Pruebas del servicio
    print("🧪 Probando Ollama Service...")
    
    try:
        service = OllamaService()
        
        # Verificar estado
        if not service._check_ollama_status():
            print("❌ Ollama no está ejecutándose. Inicia Ollama primero.")
            exit(1)
        
        print(f"✅ Ollama disponible en {service.base_url}")
        print(f"📦 Modelos disponibles: {service.available_models}")
        
        # Test básico
        if service.available_models:
            model_to_test = service.available_models[0]
            print(f"\n🧪 Probando modelo: {model_to_test}")
            
            test_result = service.test_model(model_to_test)
            print(f"Resultado: {test_result}")
            
            if test_result["disponible"]:
                # Test con contexto RAG simulado
                fragmentos_test = [
                    {"texto": "Madrid es la capital de España.", "fuente": "documentos"},
                    {"texto": "Madrid tiene más de 3 millones de habitantes.", "fuente": "web"}
                ]
                
                resultado = service.get_local_response(
                    "¿Cuántos habitantes tiene la capital?",
                    context_fragments=fragmentos_test,
                    model=model_to_test
                )
                
                print(f"\n📝 Test con RAG:")
                print(f"Respuesta: {resultado['respuesta']}")
                print(f"Tiempo: {resultado['tiempo_respuesta']}s")
                print(f"Tokens estimados: {resultado['tokens_estimados']}")
        else:
            print("⚠️ No hay modelos disponibles. Descarga alguno con: ollama pull llama3.1")
            
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")

def _check_ollama_status(self) -> bool:
    """Verifica si Ollama está ejecutándose"""
    try:
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        logger.warning("⚠️ Ollama no está disponible. Instala y ejecuta 'ollama serve'")
        return False