"""
bot_openai_langchain.py - Servicio OpenAI usando LangChain
Migración del TFM a LangChain para mejor arquitectura
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback
# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainOpenAIService:
    """Servicio OpenAI mejorado usando LangChain"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("⚠️ OPENAI_API_KEY no encontrada en variables de entorno")
        
        # Inicializar modelo con LangChain
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-4",
            temperature=0.7,
            max_tokens=512
        )
        
    def get_response(self, 
                    prompt_usuario: str,
                    context_fragments: List[Dict] = None,
                    temperature: float = 0.7,
                    max_tokens: int = 512,
                    **kwargs) -> Dict[str, Any]:
        """
        Genera respuesta usando LangChain con métricas detalladas
        """
        start_time = time.time()
        
        try:
            # Actualizar configuración del modelo
            self.llm.temperature = temperature
            self.llm.max_tokens = max_tokens
            
            # Construir mensajes
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(prompt_usuario, context_fragments)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Ejecutar con callback para métricas
            with get_openai_callback() as cb:
                response = self.llm.invoke(messages)
            
            end_time = time.time()
            
            # Construir resultado con métricas de LangChain
            resultado = {
                "respuesta": response.content.strip(),
                "modelo": self.llm.model_name,
                "tiempo_respuesta": round(end_time - start_time, 2),
                "tokens_usados": {
                    "prompt": cb.prompt_tokens,
                    "completion": cb.completion_tokens,
                    "total": cb.total_tokens
                },
                "coste_usd": cb.total_cost,
                "parametros": {
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                "fragmentos_utilizados": len(context_fragments) if context_fragments else 0,
                "exito": True,
                "error": None,
                "framework": "langchain"
            }
            
            logger.info(f"✅ LangChain OpenAI - Tokens: {cb.total_tokens}, Coste: ${cb.total_cost:.4f}")
            return resultado
            
        except Exception as e:
            error_msg = f"❌ Error LangChain OpenAI: {str(e)}"
            logger.error(error_msg)
            
            return {
                "respuesta": f"⚠️ Error: {str(e)}",
                "modelo": self.llm.model_name if hasattr(self, 'llm') else "unknown",
                "tiempo_respuesta": round(time.time() - start_time, 2),
                "tokens_usados": {"prompt": 0, "completion": 0, "total": 0},
                "coste_usd": 0.0,
                "parametros": {"temperature": temperature, "max_tokens": max_tokens},
                "fragmentos_utilizados": 0,
                "exito": False,
                "error": str(e),
                "framework": "langchain"
            }
    
    def _build_system_prompt(self) -> str:
        """Prompt del sistema optimizado para administración local"""
        return """Eres un asistente especializado en administración local española usando LangChain.

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

    def _build_user_prompt(self, pregunta: str, fragmentos: List[Dict] = None) -> str:
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
        """Prueba la conexión con OpenAI usando LangChain"""
        try:
            test_message = [HumanMessage(content="Test")]
            with get_openai_callback() as cb:
                response = self.llm.invoke(test_message)
            
            return {
                "conectado": True,
                "modelo_disponible": self.llm.model_name,
                "mensaje": "✅ Conexión exitosa con OpenAI via LangChain",
                "tokens_test": cb.total_tokens,
                "coste_test": cb.total_cost
            }
        except Exception as e:
            return {
                "conectado": False,
                "modelo_disponible": None,
                "mensaje": f"❌ Error de conexión: {str(e)}"
            }

    def change_model(self, model_name: str) -> bool:
        """Cambiar modelo dinámicamente"""
        try:
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                model=model_name,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens
            )
            logger.info(f"✅ Modelo cambiado a: {model_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Error cambiando modelo: {e}")
            return False


# Funciones de compatibilidad con el código existente
def get_openai_response(prompt_usuario: str, **kwargs) -> str:
    """Función de compatibilidad que devuelve solo el texto de respuesta"""
    try:
        service = LangChainOpenAIService()
        resultado = service.get_response(prompt_usuario, **kwargs)
        return resultado["respuesta"]
    except Exception as e:
        return f"⚠️ Error OpenAI: {str(e)}"


def get_detailed_openai_response(prompt_usuario: str, 
                                context_fragments: List[Dict] = None,
                                **kwargs) -> Dict[str, Any]:
    """Función para obtener respuesta detallada con métricas LangChain"""
    try:
        service = LangChainOpenAIService()
        return service.get_response(prompt_usuario, context_fragments, **kwargs)
    except Exception as e:
        logger.error(f"Error en get_detailed_openai_response: {e}")
        return {
            "respuesta": f"⚠️ Error: {str(e)}",
            "exito": False,
            "error": str(e),
            "framework": "langchain"
        }


if __name__ == "__main__":
    # Pruebas del servicio LangChain
    print("🧪 Probando LangChain OpenAI Service...")
    
    try:
        service = LangChainOpenAIService()
        
        # Test de conexión
        conexion = service.test_connection()
        print(f"Conexión: {conexion}")
        
        # Test de respuesta simple
        pregunta = "¿Cuál es la capital de España?"
        resultado = service.get_response(pregunta)
        print(f"\nPregunta: {pregunta}")
        print(f"Respuesta: {resultado['respuesta']}")
        print(f"Tokens usados: {resultado['tokens_usados']['total']}")
        print(f"Coste: ${resultado['coste_usd']:.4f}")
        print(f"Tiempo: {resultado['tiempo_respuesta']}s")
        
        # Test con contexto RAG simulado
        fragmentos_simulados = [
            {"texto": "Madrid es la capital de España desde 1561.", "fuente": "documentos"},
            {"texto": "La población de Madrid es de aproximadamente 3.3 millones.", "fuente": "web"}
        ]
        
        resultado_rag = service.get_response(
            "¿Cuántos habitantes tiene la capital?", 
            context_fragments=fragmentos_simulados
        )
        print(f"\nCon contexto RAG:")
        print(f"Respuesta: {resultado_rag['respuesta']}")
        print(f"Framework: {resultado_rag['framework']}")
        
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")