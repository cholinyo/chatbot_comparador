"""
Chat routes - Actualizado para ChromaDB + LlamaIndex
Mejoras en búsqueda y filtrado por tipo de documento
"""
from flask import Blueprint, render_template, request, jsonify
import logging
from datetime import datetime

# Imports actualizados
from app.utils.rag_utils_updated import (
    buscar_fragmentos_combinados,
    buscar_por_tipo_documento,
    obtener_tipos_documento_disponibles,
    buscar_fragmentos_avanzado
)
from app.services.bot_openai import get_openai_response
from app.services.bot_local import get_local_response

chat_bp = Blueprint("chat", __name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@chat_bp.route("/chat", methods=["GET", "POST"])
def chat():
    """Chat principal - Mejorado con filtros avanzados"""
    respuesta = None
    fragmentos = []
    pregunta = ""
    tipo_documento_filtro = None
    tiempo_respuesta = None
    
    # Obtener tipos de documento disponibles para filtros
    try:
        tipos_disponibles = obtener_tipos_documento_disponibles()
    except Exception as e:
        logger.error(f"❌ Error obteniendo tipos de documento: {e}")
        tipos_disponibles = []

    if request.method == "POST":
        pregunta = request.form.get("pregunta", "").strip()
        tipo_documento_filtro = request.form.get("tipo_documento", None)
        busqueda_avanzada = request.form.get("busqueda_avanzada", False)
        k_fragmentos = int(request.form.get("k_fragmentos", 5))
        
        if not pregunta:
            return render_template(
                "chat_updated.html", 
                tipos_disponibles=tipos_disponibles,
                error="Por favor, introduce una pregunta"
            )

        try:
            inicio = datetime.now()
            
            # Realizar búsqueda según filtros
            if tipo_documento_filtro and tipo_documento_filtro != "todos":
                logger.info(f"🔍 Búsqueda específica: {tipo_documento_filtro}")
                fragmentos = buscar_por_tipo_documento(
                    pregunta, 
                    tipo_documento_filtro, 
                    k=k_fragmentos
                )
            elif busqueda_avanzada:
                logger.info("🔍 Búsqueda avanzada activada")
                fragmentos = buscar_fragmentos_avanzado(
                    pregunta,
                    filtros_avanzados=None,
                    incluir_similares=True
                )
            else:
                logger.info("🔍 Búsqueda estándar")
                fragmentos = buscar_fragmentos_combinados(
                    pregunta, 
                    k=k_fragmentos
                )

            # Construir contexto para el modelo
            if fragmentos:
                contexto = "\n".join([
                    f"[{f.get('fuente', 'general')}] {f['texto']}" 
                    for f in fragmentos
                ])
                
                prompt = f"""Usa la siguiente información para responder a la pregunta de forma precisa y completa:

{contexto}

Pregunta: {pregunta}

Instrucciones:
- Basa tu respuesta únicamente en la información proporcionada
- Si la información es insuficiente, indícalo claramente
- Cita las fuentes cuando sea relevante
- Sé conciso pero completo

Respuesta:"""
            else:
                prompt = f"""No se encontró información específica para responder: "{pregunta}"

Por favor, reformula tu pregunta o verifica que exista información sobre este tema en la base de conocimiento."""

            # Generar respuesta (usar modelo local por defecto)
            respuesta = get_local_response(prompt)
            
            tiempo_respuesta = (datetime.now() - inicio).total_seconds()
            
            logger.info(f"✅ Chat completado en {tiempo_respuesta:.2f}s con {len(fragmentos)} fragmentos")
            
        except Exception as e:
            logger.error(f"❌ Error en chat: {e}")
            respuesta = f"Error procesando la consulta: {str(e)}"
            fragmentos = []
            tiempo_respuesta = None

    return render_template(
        "chat_updated.html",
        pregunta=pregunta,
        respuesta=respuesta,
        contexto=fragmentos,
        tipos_disponibles=tipos_disponibles,
        tipo_seleccionado=tipo_documento_filtro,
        tiempo_respuesta=tiempo_respuesta
    )

@chat_bp.route("/api/chat", methods=["POST"])
def api_chat():
    """API endpoint para chat programático"""
    try:
        data = request.get_json()
        
        pregunta = data.get("pregunta", "").strip()
        tipo_documento = data.get("tipo_documento", None)
        k = data.get("k", 5)
        
        if not pregunta:
            return jsonify({
                "success": False,
                "error": "Pregunta requerida"
            }), 400
        
        inicio = datetime.now()
        
        # Búsqueda
        if tipo_documento and tipo_documento != "todos":
            fragmentos = buscar_por_tipo_documento(pregunta, tipo_documento, k=k)
        else:
            fragmentos = buscar_fragmentos_combinados(pregunta, k=k)
        
        # Generar respuesta
        if fragmentos:
            contexto = "\n".join([f"[{f.get('fuente', 'general')}] {f['texto']}" for f in fragmentos])
            prompt = f"Contexto:\n{contexto}\n\nPregunta: {pregunta}\nRespuesta:"
        else:
            prompt = f"No se encontró información para: {pregunta}"
        
        respuesta = get_local_response(prompt)
        tiempo = (datetime.now() - inicio).total_seconds()
        
        return jsonify({
            "success": True,
            "respuesta": respuesta,
            "fragmentos_count": len(fragmentos),
            "tiempo_respuesta": tiempo,
            "fragmentos": [
                {
                    "texto": f["texto"][:200] + "..." if len(f["texto"]) > 200 else f["texto"],
                    "fuente": f.get("fuente", "general"),
                    "metadata": f.get("metadata", {})
                }
                for f in fragmentos[:3]  # Solo primeros 3 para API
            ]
        })
        
    except Exception as e:
        logger.error(f"❌ Error en API chat: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@chat_bp.route("/api/tipos_documento", methods=["GET"])
def api_tipos_documento():
    """API para obtener tipos de documento disponibles"""
    try:
        tipos = obtener_tipos_documento_disponibles()
        return jsonify({
            "success": True,
            "tipos": tipos
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@chat_bp.route("/chat/avanzado", methods=["GET", "POST"])
def chat_avanzado():
    """Interfaz de chat avanzado con más opciones"""
    if request.method == "GET":
        tipos_disponibles = obtener_tipos_documento_disponibles()
        return render_template(
            "chat_avanzado.html",
            tipos_disponibles=tipos_disponibles
        )
    
    # POST - Procesar consulta avanzada
    try:
        pregunta = request.form.get("pregunta", "").strip()
        filtros = {}
        
        # Construir filtros avanzados
        if request.form.get("tipo_documento"):
            filtros["document_type"] = request.form.get("tipo_documento")
        
        if request.form.get("fecha_desde"):
            filtros["fecha_desde"] = request.form.get("fecha_desde")
        
        # Parámetros de búsqueda
        k = int(request.form.get("k_fragmentos", 10))
        incluir_similares = bool(request.form.get("incluir_similares"))
        
        # Búsqueda avanzada
        fragmentos = buscar_fragmentos_avanzado(
            pregunta,
            filtros_avanzados=filtros,
            incluir_similares=incluir_similares
        )
        
        # Generar respuesta
        if fragmentos:
            contexto = "\n".join([f"[{f.get('fuente', 'general')}] {f['texto']}" for f in fragmentos])
            prompt = f"""Información disponible:
{contexto}

Pregunta: {pregunta}

Proporciona una respuesta detallada basada únicamente en la información anterior:"""
            
            respuesta = get_local_response(prompt)
        else:
            respuesta = "No se encontró información relevante para tu consulta."
        
        return render_template(
            "chat_avanzado.html",
            pregunta=pregunta,
            respuesta=respuesta,
            fragmentos=fragmentos,
            tipos_disponibles=obtener_tipos_documento_disponibles(),
            filtros_aplicados=filtros
        )
        
    except Exception as e:
        logger.error(f"❌ Error en chat avanzado: {e}")
        return render_template(
            "chat_avanzado.html",
            error=f"Error: {str(e)}",
            tipos_disponibles=obtener_tipos_documento_disponibles()
        )