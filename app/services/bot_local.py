"""
bot_local_langchain.py - Servicio modelos locales usando LangChain
Migración del TFM a LangChain para mejor arquitectura
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional

from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenCounterCallback(BaseCallbackHandler):
    """Callback personalizado para contar tokens en modelos locales"""
    
    def __init__(self):
        self.token_count = 0
        self.start_time = None
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()
        self.token_count = 0
        
    def on_llm_new_token(self, token: str, **kwargs):
        self.token_count += 1

class LangChainOllamaService:
    """Servicio Ollama mejorado usando LangChain"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.default_model = "llama3.1:8b"
        self.current_model = self.default_model
        
        # Inicializar modelo con LangChain
        self.llm = ChatOllama(
            model=self.default_model,
            base_url=base_url,
            temperature=0.3
        )
        
        # Verificar que Ollama esté funcionando
        if not self._check_ollama_status():
            logger.warning("⚠️ Ollama no está disponible. Algunas funciones pueden fallar.")
    
    def _check_ollama_status(self) -> bool:
        """Verifica si Ollama está ejecutándose"""
        try:
            # Intentar una consulta simple
            test_message = [HumanMessage(content="test")]
            self.llm.invoke(test_message)
            return True
        except Exception:
            return False
    
    def get_response(self,
                    prompt_usuario: str,
                    context_fragments: List[Dict] = None,
                    model: str = None,
                    temperature: float = 0.3,
                    **kwargs) -> Dict[str, Any]:
        """
        Genera respuesta usando LangChain Ollama con métricas
        """
        start_time = time.time()
        model_to_use = model or self.current_model
        
        try:
            # Actualizar modelo si es necesario
            if model_to_use != self.current_model:
                self._change_model(model_to_use)
            
            # Actualizar configuración
            self.llm.temperature = temperature
            
            # Construir mensajes
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(prompt_usuario, context_fragments)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Ejecutar con callback para métricas
            callback = TokenCounterCallback()
            response = self.llm.invoke(messages, callbacks=[callback])
            
            end_time = time.time()
            
            # Construir resultado
            resultado = {
                "respuesta": response.content.strip(),
                "modelo": model_to_use,
                "tiempo_respuesta": round(end_time - start_time, 2),
                "tokens_estimados": callback.token_count,
                "parametros": {
                    "temperature": temperature,
                    "base_url": self.base_url
                },
                "fragmentos_utilizados": len(context_fragments) if context_fragments else 0,
                "exito": True,
                "error": None,
                "framework": "langchain"
            }
            
            logger.info(f"✅ LangChain Ollama - Modelo: {model_to_use}, Tiempo: {resultado['tiempo_respuesta']}s")
            return resultado
            
        except Exception as e:
            error_msg = f"❌ Error LangChain Ollama: {str(e)}"
            logger.error(error_msg)
            
            return {
                "respuesta": f"⚠️ Error en modelo local: {str(e)}",
                "modelo": model_to_use,
                "tiempo_respuesta": round(time.time() - start_time, 2),
                "tokens_estimados": 0,
                "parametros": {"temperature": temperature},
                "fragmentos_utilizados": 0,
                "exito": False,
                "error": str(e),
                "framework": "langchain"
            }
    
    def _build_system_prompt(self) -> str:
        """Prompt del sistema para administración local"""
        return """Eres un asistente especializado en administración local española usando LangChain y Ollama.

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

    def _change_model(self, model_name: str) -> bool:
        """Cambiar modelo dinámicamente"""
        try:
            self.llm = ChatOllama(
                model=model_name,
                base_url=self.base_url,
                temperature=self.llm.temperature
            )
            self.current_model = model_name
            logger.info(f"✅ Modelo cambiado a: {model_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Error cambiando modelo: {e}")
            return False
    
    def test_model(self, model: str = None) -> Dict[str, Any]:
        """Prueba un modelo específico"""
        model_to_test = model or self.current_model
        
        try:
            if model_to_test != self.current_model:
                self._change_model(model_to_test)
            
            test_prompt = "¿Cuál es la capital de España? Responde brevemente."
            resultado = self.get_response(test_prompt)
            
            return {
                "modelo": model_to_test,
                "disponible": resultado["exito"],
                "tiempo_respuesta": resultado["tiempo_respuesta"],
                "mensaje": "✅ Modelo funcionando correctamente" if resultado["exito"] else "❌ Error en el modelo",
                "framework": "langchain"
            }
            
        except Exception as e:
            return {
                "modelo": model_to_test,
                "disponible": False,
                "tiempo_respuesta": 0,
                "mensaje": f"❌ Error: {str(e)}",
                "framework": "langchain"
            }
    
    def get_available_models(self) -> List[str]:
        """Obtiene lista de modelos disponibles"""
        try:
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            return []
        except Exception as e:
            logger.error(f"Error obteniendo modelos: {e}")
            return ["llama3.1:8b", "mistral:7b", "gemma:7b"]  # Fallback


# Funciones de compatibilidad con el código existente
def get_local_response(prompt_usuario: str, **kwargs) -> str:
    """Función de compatibilidad que devuelve solo el texto"""
    try:
        service = LangChainOllamaService()
        resultado = service.get_response(prompt_usuario, **kwargs)
        return resultado["respuesta"]
    except Exception as e:
        logger.error(f"Error en get_local_response: {e}")
        return f"⚠️ Error modelo local: {str(e)}"


def get_detailed_local_response(prompt_usuario: str,
                               context_fragments: List[Dict] = None,
                               **kwargs) -> Dict[str, Any]:
    """Función para obtener respuesta detallada con métricas LangChain"""
    try:
        service = LangChainOllamaService()
        return service.get_response(prompt_usuario, context_fragments, **kwargs)
    except Exception as e:
        logger.error(f"Error en get_detailed_local_response: {e}")
        return {
            "respuesta": f"⚠️ Error: {str(e)}",
            "exito": False,
            "error": str(e),
            "framework": "langchain"
        }


if __name__ == "__main__":
    # Pruebas del servicio LangChain
    print("🧪 Probando LangChain Ollama Service...")
    
    try:
        service = LangChainOllamaService()
        
        # Verificar estado
        if not service._check_ollama_status():
            print("❌ Ollama no está ejecutándose. Inicia Ollama primero.")
            exit(1)
        
        print(f"✅ Ollama disponible en {service.base_url}")
        
        # Obtener modelos disponibles
        modelos = service.get_available_models()
        print(f"📦 Modelos disponibles: {modelos}")
        
        # Test básico
        if modelos:
            model_to_test = modelos[0]
            print(f"\n🧪 Probando modelo: {model_to_test}")
            
            test_result = service.test_model(model_to_test)
            print(f"Resultado: {test_result}")
            
            if test_result["disponible"]:
                # Test con contexto RAG simulado
                fragmentos_test = [
                    {"texto": "Madrid es la capital de España.", "fuente": "documentos"},
                    {"texto": "Madrid tiene más de 3 millones de habitantes.", "fuente": "web"}
                ]
                
                resultado = service.get_response(
                    "¿Cuántos habitantes tiene la capital?",
                    context_fragments=fragmentos_test,
                    model=model_to_test
                )
                
                print(f"\n📝 Test con RAG:")
                print(f"Respuesta: {resultado['respuesta']}")
                print(f"Tiempo: {resultado['tiempo_respuesta']}s")
                print(f"Tokens estimados: {resultado['tokens_estimados']}")
                print(f"Framework: {resultado['framework']}")
        else:
            print("⚠️ No hay modelos disponibles. Descarga alguno con: ollama pull llama3.1")
            
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")