import logging
from flask import Blueprint, render_template, request
from app.utils.rag_utils import buscar_fragmentos_combinados
from app.services.model_manager import model_manager

logger = logging.getLogger(__name__)
comparador_bp = Blueprint("comparador", __name__)

@comparador_bp.route("/comparar", methods=["GET", "POST"])
def comparar():
    resultado_local = None
    resultado_openai = None
    fragmentos = []
    pregunta = ""
    error_local = None
    error_openai = None

    if request.method == "POST":
        pregunta = request.form.get("pregunta")
        modelo_local = request.form.get("modelo_local", "local")  # local, ollama:llama3.2, etc.
        modelo_openai = request.form.get("modelo_openai", "openai:gpt-4")  # openai:gpt-4, etc.
        
        logger.info(f"🔵 COMPARADOR: Pregunta - Local: {modelo_local}, OpenAI: {modelo_openai}")
        
        # Comparar modelos con RAG integrado
        modelos_a_comparar = []
        
        # Solo añadir modelo local si se solicita
        if modelo_local and modelo_local != "none":
            modelos_a_comparar.append(modelo_local)
        
        # Solo añadir OpenAI si se solicita explícitamente
        if modelo_openai and modelo_openai != "none":
            modelos_a_comparar.append(modelo_openai)
        
        if modelos_a_comparar:
            logger.info(f"🔀 COMPARADOR: Comparando modelos con RAG: {modelos_a_comparar}")
            
            # Comparar con RAG integrado
            resultados = {}
            for modelo in modelos_a_comparar:
                logger.info(f"🔀 Procesando modelo: {modelo}")
                resultado = model_manager.get_response(
                    prompt=pregunta,  # Pregunta original
                    model_type=modelo,
                    use_rag=True,  # IMPORTANTE: RAG habilitado
                    question=pregunta,  # Para búsqueda de fragmentos
                    rag_k=3  # Menos fragmentos para comparación más rápida
                )
                resultados[modelo] = resultado
                
                # Usar fragmentos del primer modelo exitoso
                if resultado["success"] and resultado["rag_used"] and not fragmentos:
                    fragmentos = resultado["rag_fragments"]
            
            # Procesar resultados
            for modelo, resultado in resultados.items():
                if modelo.startswith("local") or modelo.startswith("ollama") or modelo.startswith("file"):
                    resultado_local = {
                        "respuesta": resultado["response"],
                        "modelo": resultado["model_used"],
                        "tiempo": resultado["time_taken"],
                        "success": resultado["success"],
                        "rag_used": resultado["rag_used"],
                        "rag_fragments": len(resultado.get("rag_fragments", []))
                    }
                    if not resultado["success"]:
                        error_local = resultado["error"]
                
                elif modelo.startswith("openai"):
                    resultado_openai = {
                        "respuesta": resultado["response"],
                        "modelo": resultado["model_used"],
                        "tiempo": resultado["time_taken"],
                        "success": resultado["success"],
                        "rag_used": resultado["rag_used"],
                        "rag_fragments": len(resultado.get("rag_fragments", []))
                    }
                    if not resultado["success"]:
                        error_openai = resultado["error"]

    # Obtener modelos disponibles
    modelos_disponibles = model_manager.get_available_models()

    return render_template("comparar.html",
                           pregunta=pregunta,
                           resultado_local=resultado_local,
                           resultado_openai=resultado_openai,
                           fragmentos=fragmentos,
                           error_local=error_local,
                           error_openai=error_openai,
                           modelos_disponibles=modelos_disponibles)