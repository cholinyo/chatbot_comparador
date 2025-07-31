import os
import logging
from dotenv import load_dotenv
import openai

logger = logging.getLogger(__name__)
load_dotenv()

# Configurar la clave API
openai.api_key = os.getenv("OPENAI_API_KEY")

def is_openai_configured():
    """Verifica si OpenAI está configurado correctamente"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("⚠️ OPENAI_API_KEY no configurada")
        return False
    
    if not api_key.startswith("sk-"):
        logger.warning("⚠️ OPENAI_API_KEY no parece válida")
        return False
    
    return True

def test_openai_connection():
    """Prueba la conexión con OpenAI"""
    if not is_openai_configured():
        return False, "API Key no configurada"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        return True, "Conexión exitosa"
    except Exception as e:
        return False, str(e)

def get_openai_response(prompt_usuario, model="gpt-4", force=False):
    """
    Genera respuesta usando OpenAI - SOLO si se solicita explícitamente
    
    Args:
        prompt_usuario (str): El prompt del usuario
        model (str): Modelo de OpenAI a usar
        force (bool): Forzar llamada incluso si no está configurado
    
    Returns:
        str: Respuesta generada o mensaje de error
    """
    
    # CONTROL ESTRICTO - Solo ejecutar si se solicita explícitamente
    if not force:
        logger.warning("🚫 get_openai_response llamada sin force=True - Bloqueada")
        return "⚠️ Llamada a OpenAI no autorizada. Use force=True si realmente desea usar OpenAI."
    
    logger.info(f"🔵 get_openai_response - AUTORIZADA con force=True - Modelo: {model}")
    
    if not is_openai_configured():
        error_msg = "❌ OpenAI no está configurado. Verifica OPENAI_API_KEY en .env"
        logger.error(error_msg)
        return error_msg

    try:
        from app.config.settings import get_openai_model
        model_to_use = model if model != "gpt-4" else get_openai_model()
        
        system_prompt = (
            "Eres un asistente especializado en administración pública. "
            "Responde de forma precisa y profesional basándote en la información proporcionada."
        )

        logger.info(f"🔵 Enviando consulta a OpenAI - Modelo: {model_to_use}")
        response = openai.ChatCompletion.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_usuario}
            ],
            max_tokens=512,
            temperature=0.7
        )
        
        respuesta = response["choices"][0]["message"]["content"].strip()
        tokens_used = response.get("usage", {}).get("total_tokens", 0)
        
        logger.info(f"✅ Respuesta OpenAI generada - Tokens: {tokens_used}, Caracteres: {len(respuesta)}")
        return respuesta
        
    except Exception as e:
        error_msg = f"❌ Error OpenAI: {e}"
        logger.error(error_msg)
        return error_msg

def get_openai_models():
    """Obtiene lista de modelos disponibles en OpenAI"""
    if not is_openai_configured():
        return []
    
    try:
        models = openai.Model.list()
        return [model.id for model in models.data if 'gpt' in model.id]
    except:
        return ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]  # Fallback por defecto