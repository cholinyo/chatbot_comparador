"""
bot_openai.py - Servicio mejorado para integración con OpenAI API
Parte del TFM: Prototipo de Chatbot Interno para Administraciones Locales
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import openai
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIService:
    """Servicio para interactuar con la API de OpenAI con funcionalidades avanzadas"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("⚠️ OPENAI_API_KEY no encontrada en variables de entorno")
        
        self.client = OpenAI(api_key=self.api_key)
        self.default_model = "gpt-4"
        self.max_retries = 3
        self.retry_delay = 1
        
    def get_openai_response(self, 
                          prompt_usuario: str,
                          context_fragments: list = None,
                          model: str = None,
                          temperature: float = 0.7,
                          max_tokens: int = 512,
                          **kwargs) -> Dict[str, Any]:
        """
        Genera respuesta usando OpenAI con contexto RAG y métricas detalladas
        
        Args:
            prompt_usuario: Pregunta del usuario
            context_fragments: Lista de fragmentos RAG recuperados
            model: Modelo a usar (por defecto gpt-4)
            temperature: Creatividad de la respuesta (0-1)
            max_tokens: Máximo de tokens en la respuesta
            **kwargs: Parámetros adicionales para la API
            
        Returns:
            Dict con respuesta, métricas y metadatos
        """
        start_time = time.time()
        
        try:
            # Construir prompt con contexto RAG
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(prompt_usuario, context_fragments)
            
            # Configurar parámetros
            model_to_use = model or self.default_model
            
            # Realizar llamada a la API
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=kwargs.get('top_p', 1.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0),
                presence_penalty=kwargs.get('presence_penalty', 0)
            )
            
            end_time = time.time()
            
            # Extraer información de la respuesta
            respuesta_texto = response.choices[0].message.content.strip()
            
            # Construir resultado con métricas
            resultado = {
                "respuesta": respuesta_texto,
                "modelo": model_to_use,
                "tiempo_respuesta": round(end_time - start_time, 2),
                "tokens_usados": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "parametros": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": kwargs.get('top_p', 1.0)
                },
                "fragmentos_utilizados": len(context_fragments) if context_fragments else 0,
                "exito": True,
                "error": None
            }
            
            logger.info(f"✅ Respuesta OpenAI generada - Tokens: {response.usage.total_tokens}, Tiempo: {resultado['tiempo_respuesta']}s")
            return resultado
            
        except Exception as e:
            error_msg = f"❌ Error OpenAI: {str(e)}"
            logger.error(error_msg)
            
            return {
                "respuesta": f"⚠️ Error al conectar con OpenAI: {str(e)}",
                "modelo": model_to_use if 'model_to_use' in locals() else "unknown",
                "tiempo_respuesta": round(time.time() - start_time, 2),
                "tokens_usados": {"prompt": 0, "completion": 0, "total": 0},
                "parametros": {"temperature": temperature, "max_tokens": max_tokens},
                "fragmentos_utilizados": 0,
                "exito": False,
                "error": str(e)
            }
    
    def _build_system_prompt(self) -> str:
        """Construye el prompt del sistema optimizado para administración local"""
        return """Eres un asistente especializado en administración local española. 

CARACTERÍSTICAS:
- Respondes basándote únicamente en la información proporcionada
- Eres preciso, claro y directo
- Citas las fuentes cuando es relevante
- Indicas si no tienes información suficiente

FORMATO DE RESPUESTA:
- Respuesta directa y concisa
- Si usas información específica, menciona la fuente
- Si hay dudas, recomienda contactar con el departamento correspondiente

LIMITACIONES:
- No inventes información que no esté en el contexto
- No hagas interpretaciones legales definitivas
- Sugiere verificación oficial cuando sea necesario"""

    def _build_user_prompt(self, pregunta: str, fragmentos: list = None) -> str:
        """Construye el prompt del usuario con contexto RAG"""
        if not fragmentos:
            return f"Pregunta: {pregunta}\n\nResponde basándote en tu conocimiento general sobre administración local."
        
        # Construir contexto con fragmentos
        contexto = "\n".join([
            f"Fuente {i+1} ({frag.get('fuente', 'desconocida')}): {frag.get('texto', '')}"
            for i, frag in enumerate(fragmentos[:5])  # Limitar a 5 fragmentos
        ])
        
        return f"""Información disponible:
{contexto}

Pregunta: {pregunta}

Responde basándote en la información proporcionada. Si la información es insuficiente, indícalo claramente."""

    def test_connection(self) -> Dict[str, Any]:
        """Prueba la conexión con OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            return {
                "conectado": True,
                "modelo_disponible": "gpt-3.5-turbo",
                "mensaje": "✅ Conexión exitosa con OpenAI"
            }
        except Exception as e:
            return {
                "conectado": False,
                "modelo_disponible": None,
                "mensaje": f"❌ Error de conexión: {str(e)}"
            }

    def get_available_models(self) -> list:
        """Obtiene lista de modelos disponibles"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data if "gpt" in model.id]
        except Exception as e:
            logger.error(f"Error obteniendo modelos: {e}")
            return ["gpt-3.5-turbo", "gpt-4"]

    def calculate_cost(self, tokens_used: dict, model: str) -> float:
        """Calcula coste aproximado de la consulta"""
        # Precios aproximados (actualizar según precios actuales)
        precios = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # por 1K tokens
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03}
        }
        
        if model not in precios:
            return 0.0
        
        precio_input = (tokens_used.get("prompt", 0) / 1000) * precios[model]["input"]
        precio_output = (tokens_used.get("completion", 0) / 1000) * precios[model]["output"]
        
        return round(precio_input + precio_output, 6)


# Función de compatibilidad con el código existente
def get_openai_response(prompt_usuario: str, **kwargs) -> str:
    """Función de compatibilidad que devuelve solo el texto de respuesta"""
    try:
        service = OpenAIService()
        resultado = service.get_openai_response(prompt_usuario, **kwargs)
        return resultado["respuesta"]
    except Exception as e:
        return f"⚠️ Error OpenAI: {str(e)}"


# Función para comparación con modelos locales
def get_detailed_openai_response(prompt_usuario: str, 
                                context_fragments: list = None,
                                **kwargs) -> Dict[str, Any]:
    """Función para obtener respuesta detallada con métricas"""
    try:
        service = OpenAIService()
        return service.get_openai_response(prompt_usuario, context_fragments, **kwargs)
    except Exception as e:
        logger.error(f"Error en get_detailed_openai_response: {e}")
        return {
            "respuesta": f"⚠️ Error: {str(e)}",
            "exito": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Pruebas del servicio
    print("🧪 Probando OpenAI Service...")
    
    try:
        service = OpenAIService()
        
        # Test de conexión
        conexion = service.test_connection()
        print(f"Conexión: {conexion}")
        
        # Test de respuesta simple
        pregunta = "¿Cuál es la capital de España?"
        resultado = service.get_openai_response(pregunta)
        print(f"\nPregunta: {pregunta}")
        print(f"Respuesta: {resultado['respuesta']}")
        print(f"Tokens usados: {resultado['tokens_usados']['total']}")
        print(f"Tiempo: {resultado['tiempo_respuesta']}s")
        
        # Test con contexto RAG simulado
        fragmentos_simulados = [
            {"texto": "Madrid es la capital de España desde 1561.", "fuente": "documentos"},
            {"texto": "La población de Madrid es de aproximadamente 3.3 millones.", "fuente": "web"}
        ]
        
        resultado_rag = service.get_openai_response(
            "¿Cuántos habitantes tiene la capital?", 
            context_fragments=fragmentos_simulados
        )
        print(f"\nCon contexto RAG:")
        print(f"Respuesta: {resultado_rag['respuesta']}")
        
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")