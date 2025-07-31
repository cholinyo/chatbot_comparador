import logging
from flask import Blueprint, render_template, request, session
from app.utils.rag_utils import buscar_fragmentos_combinados
from app.services.model_manager import model_manager

logger = logging.getLogger(__name__)
chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["GET", "POST"])
def chat():
    respuesta = None
    fragmentos = []
    pregunta = ""
    modelo_usado = None
    tiempo_respuesta = None
    error = None

    if request.method == "POST":
        pregunta = request.form.get("pregunta")
        modelo_seleccionado = request.form.get("modelo", "local")  # Por defecto: local
        
        logger.info(f"🔵 CHAT: Pregunta recibida - Modelo: {modelo_seleccionado}")
        
        # Generar respuesta usando el modelo seleccionado CON RAG
        try:
            resultado = model_manager.get_response(
                prompt=pregunta,  # La pregunta original como prompt base
                model_type=modelo_seleccionado,
                use_rag=True,  # IMPORTANTE: Habilitar RAG
                question=pregunta,  # La pregunta para buscar fragmentos
                rag_k=5  # Número de fragmentos a recuperar
            )
            
            if resultado["success"]:
                respuesta = resultado["response"]
                modelo_usado = resultado["model_used"]
                tiempo_respuesta = round(resultado["time_taken"], 2)
                
                # Usar fragmentos del RAG integrado
                if resultado["rag_used"]:
                    fragmentos = resultado["rag_fragments"]
                    logger.info(f"✅ CHAT: Respuesta con RAG - {modelo_usado} - {len(fragmentos)} fragmentos - {tiempo_respuesta}s")
                else:
                    logger.info(f"✅ CHAT: Respuesta sin RAG - {modelo_usado} - {tiempo_respuesta}s")
            else:
                error = resultado["error"]
                respuesta = f"Error al generar respuesta: {error}"
                logger.error(f"❌ CHAT: Error generando respuesta: {error}")
                
        except Exception as e:
            error = str(e)
            respuesta = f"Error inesperado: {error}"
            logger.error(f"❌ CHAT: Error inesperado: {e}")

    # Obtener modelos disponibles para el selector
    modelos_disponibles = model_manager.get_available_models()
    
    return render_template("chat.html", 
                         pregunta=pregunta, 
                         respuesta=respuesta, 
                         contexto=fragmentos,
                         modelo_usado=modelo_usado,
                         tiempo_respuesta=tiempo_respuesta,
                         error=error,
                         modelos_disponibles=modelos_disponibles)

@chat_bp.route("/chat/status")
def chat_status():
    """Endpoint para verificar el estado de los modelos"""
    status = model_manager.get_system_status()
    return render_template("chat_status.html", status=status)