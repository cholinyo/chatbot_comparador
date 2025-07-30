# app/routes/chat.py - Versión mejorada con métricas
from flask import Blueprint, render_template, request, jsonify, session
from app.utils.rag_utils import buscar_fragmentos_combinados, obtener_tipos_documento_disponibles
from app.services.bot_openai import get_openai_response
from app.services.bot_local import get_local_response
from app.utils.metrics_evaluator import MetricsEvaluator, TFM_TEST_QUERIES
import time
import json
from datetime import datetime

chat_bp = Blueprint("chat", __name__)

# Inicializar evaluador de métricas
metrics_evaluator = MetricsEvaluator()

@chat_bp.route("/chat", methods=["GET", "POST"])
def chat():
    """Chat mejorado con métricas y filtros avanzados"""
    respuesta = None
    fragmentos = []
    pregunta = ""
    metrics = None
    tipos_disponibles = obtener_tipos_documento_disponibles()
    
    if request.method == "POST":
        pregunta = request.form.get("pregunta")
        tipo_filtro = request.form.get("tipo_documento", "")
        num_fragmentos = int(request.form.get("num_fragmentos", 5))
        modelo_seleccionado = request.form.get("modelo", "local")
        mostrar_metricas = request.form.get("mostrar_metricas") == "on"
        
        # Preparar filtros
        filtros = {}
        if tipo_filtro and tipo_filtro != "todos":
            filtros["document_type"] = tipo_filtro
        
        start_time = time.time()
        
        # Recuperar fragmentos con filtros
        fragmentos = buscar_fragmentos_combinados(
            pregunta, 
            k=num_fragmentos, 
            filtros=filtros if filtros else None
        )
        
        # Construir contexto
        contexto = "\n".join([f"- {f['texto']}" for f in fragmentos])
        prompt = f"""Usa la siguiente información para responder a la pregunta:

{contexto}

Pregunta: {pregunta}
Respuesta:"""
        
        # Generar respuesta según modelo seleccionado
        if modelo_seleccionado == "openai":
            respuesta = get_openai_response(prompt)
            model_name = "OpenAI GPT-4"
        else:
            respuesta = get_local_response(prompt)
            model_name = "Modelo Local"
        
        # Medir rendimiento si se solicita
        if mostrar_metricas:
            end_time = time.time()
            metrics = {
                "latency": round(end_time - start_time, 3),
                "fragments_count": len(fragmentos),
                "model_used": model_name,
                "document_types": list(set([f.get('document_type', 'unknown') for f in fragmentos])),
                "sources": list(set([f.get('fuente', 'unknown') for f in fragmentos])),
                "query_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Guardar métricas para TFM
            session['last_query_metrics'] = metrics
    
    return render_template("chat.html", 
                         pregunta=pregunta, 
                         respuesta=respuesta, 
                         contexto=fragmentos,
                         tipos_disponibles=tipos_disponibles,
                         metrics=metrics)

@chat_bp.route("/chat/compare", methods=["POST"])
def compare_models():
    """Endpoint para comparación directa de modelos"""
    data = request.get_json()
    pregunta = data.get("pregunta", "")
    
    if not pregunta:
        return jsonify({"error": "Pregunta requerida"}), 400
    
    # Usar el evaluador de métricas para comparación
    def rag_function(query, k=5):
        return buscar_fragmentos_combinados(query, k=k)
    
    try:
        comparison = metrics_evaluator.compare_models(
            pregunta,
            get_openai_response,
            get_local_response,
            rag_function
        )
        
        return jsonify({
            "comparison_id": f"{comparison.openai_metrics.query_id}_{comparison.local_metrics.query_id}",
            "openai": {
                "response": comparison.openai_metrics.response_text,
                "latency": comparison.openai_metrics.latency_seconds,
                "fragments": comparison.openai_metrics.fragments_retrieved,
                "confidence": comparison.openai_metrics.confidence_score
            },
            "local": {
                "response": comparison.local_metrics.response_text,
                "latency": comparison.local_metrics.latency_seconds,
                "fragments": comparison.local_metrics.fragments_retrieved,
                "confidence": comparison.local_metrics.confidence_score
            },
            "metadata": {
                "query": pregunta,
                "timestamp": datetime.now().isoformat(),
                "document_types": comparison.openai_metrics.document_types_used
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@chat_bp.route("/chat/metrics")
def get_metrics():
    """Endpoint para obtener métricas del sistema"""
    
    openai_summary = metrics_evaluator.get_performance_summary("openai")
    local_summary = metrics_evaluator.get_performance_summary("local")
    document_usage = metrics_evaluator.get_document_type_usage()
    
    return jsonify({
        "performance": {
            "openai": openai_summary,
            "local": local_summary
        },
        "document_usage": document_usage,
        "last_updated": datetime.now().isoformat()
    })

@chat_bp.route("/chat/test-suite")
def run_test_suite():
    """Ejecuta suite de pruebas predefinidas para TFM"""
    
    results = []
    
    for i, query in enumerate(TFM_TEST_QUERIES[:5]):  # Primeras 5 para no sobrecargar
        try:
            def rag_function(q, k=5):
                return buscar_fragmentos_combinados(q, k=k)
            
            comparison = metrics_evaluator.compare_models(
                query,
                get_openai_response,
                get_local_response,
                rag_function
            )
            
            results.append({
                "query_number": i + 1,
                "query": query,
                "openai_latency": comparison.openai_metrics.latency_seconds,
                "local_latency": comparison.local_metrics.latency_seconds,
                "fragments_retrieved": comparison.openai_metrics.fragments_retrieved,
                "document_types": comparison.openai_metrics.document_types_used
            })
            
        except Exception as e:
            results.append({
                "query_number": i + 1,
                "query": query,
                "error": str(e)
            })
    
    return jsonify({
        "test_results": results,
        "summary": {
            "total_tests": len(results),
            "successful_tests": len([r for r in results if "error" not in r]),
            "execution_timestamp": datetime.now().isoformat()
        }
    })

@chat_bp.route("/chat/export-tfm")
def export_tfm_data():
    """Exporta datos para TFM"""
    
    export_data = metrics_evaluator.export_metrics_for_tfm()
    
    return jsonify({
        "message": "Datos exportados correctamente",
        "file_path": "tfm_metrics_export.json",
        "data_summary": {
            "total_queries": export_data["system_performance"]["overall"]["total_queries"],
            "models_compared": ["openai", "local"],
            "document_types_analyzed": len(export_data["document_type_analysis"]),
            "export_timestamp": export_data["export_timestamp"]
        },
        "download_ready": True
    })
@chat_bp.route("/chat/test-suite", methods=["GET"])
def test_suite():
    # Tu lógica aquí
    return jsonify({"message": "Test suite executed"})