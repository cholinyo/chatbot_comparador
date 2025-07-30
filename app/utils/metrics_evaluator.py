# app/utils/metrics_evaluator.py
"""
Sistema de métricas avanzadas para evaluación de rendimiento
Comparación OpenAI vs Modelos Locales con métricas detalladas
"""

import time
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import hashlib
import os

@dataclass
class QueryMetrics:
    """Métricas de una consulta individual"""
    query_id: str
    timestamp: datetime
    query_text: str
    model_name: str
    response_text: str
    latency_seconds: float
    tokens_used: Optional[int]
    fragments_retrieved: int
    fragments_sources: List[str]
    document_types_used: List[str]
    confidence_score: Optional[float]
    error_occurred: bool
    error_message: Optional[str]

@dataclass
class ComparisonResult:
    """Resultado de comparación entre modelos"""
    openai_metrics: QueryMetrics
    local_metrics: QueryMetrics
    preference_score: Optional[float]  # 1=OpenAI mejor, 0=local mejor, 0.5=empate
    evaluator_notes: Optional[str]

class MetricsEvaluator:
    """Evaluador de métricas para el sistema RAG"""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Inicializa la base de datos de métricas"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de métricas individuales
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_metrics (
                query_id TEXT PRIMARY KEY,
                timestamp TEXT,
                query_text TEXT,
                model_name TEXT,
                response_text TEXT,
                latency_seconds REAL,
                tokens_used INTEGER,
                fragments_retrieved INTEGER,
                fragments_sources TEXT,
                document_types_used TEXT,
                confidence_score REAL,
                error_occurred BOOLEAN,
                error_message TEXT
            )
        ''')
        
        # Tabla de comparaciones
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comparisons (
                comparison_id TEXT PRIMARY KEY,
                timestamp TEXT,
                query_text TEXT,
                openai_query_id TEXT,
                local_query_id TEXT,
                preference_score REAL,
                evaluator_notes TEXT,
                FOREIGN KEY (openai_query_id) REFERENCES query_metrics (query_id),
                FOREIGN KEY (local_query_id) REFERENCES query_metrics (query_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_query_id(self, query_text: str, model_name: str) -> str:
        """Genera ID único para una consulta"""
        content = f"{query_text}_{model_name}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def measure_query_performance(self, 
                                 query_text: str, 
                                 model_name: str,
                                 rag_function,
                                 llm_function) -> QueryMetrics:
        """Mide el rendimiento de una consulta completa"""
        
        query_id = self.generate_query_id(query_text, model_name)
        start_time = time.time()
        error_occurred = False
        error_message = None
        
        try:
            # Recuperar fragmentos RAG
            fragmentos = rag_function(query_text, k=5)
            fragments_retrieved = len(fragmentos)
            fragments_sources = [f.get('fuente', 'unknown') for f in fragmentos]
            document_types_used = list(set([f.get('document_type', 'unknown') for f in fragmentos]))
            
            # Construir prompt con contexto
            contexto = "\n".join([f"- {f['texto']}" for f in fragmentos])
            prompt = f"""Usa la siguiente información para responder a la pregunta:

{contexto}

Pregunta: {query_text}
Respuesta:"""
            
            # Generar respuesta
            response_text = llm_function(prompt)
            
            # Calcular confianza (aproximación basada en fragmentos)
            confidence_score = min(1.0, fragments_retrieved / 5.0) * 0.8
            
            tokens_used = None  # Se calculará específicamente para OpenAI
            
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            response_text = f"Error: {error_message}"
            fragments_retrieved = 0
            fragments_sources = []
            document_types_used = []
            confidence_score = 0.0
            tokens_used = None
        
        end_time = time.time()
        latency_seconds = end_time - start_time
        
        metrics = QueryMetrics(
            query_id=query_id,
            timestamp=datetime.now(),
            query_text=query_text,
            model_name=model_name,
            response_text=response_text,
            latency_seconds=latency_seconds,
            tokens_used=tokens_used,
            fragments_retrieved=fragments_retrieved,
            fragments_sources=fragments_sources,
            document_types_used=document_types_used,
            confidence_score=confidence_score,
            error_occurred=error_occurred,
            error_message=error_message
        )
        
        self._save_metrics(metrics)
        return metrics
    
    def _save_metrics(self, metrics: QueryMetrics):
        """Guarda métricas en la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO query_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            metrics.query_id,
            metrics.timestamp.isoformat(),
            metrics.query_text,
            metrics.model_name,
            metrics.response_text,
            metrics.latency_seconds,
            metrics.tokens_used,
            metrics.fragments_retrieved,
            json.dumps(metrics.fragments_sources),
            json.dumps(metrics.document_types_used),
            metrics.confidence_score,
            metrics.error_occurred,
            metrics.error_message
        ))
        
        conn.commit()
        conn.close()
    
    def compare_models(self, 
                      query_text: str,
                      openai_function,
                      local_function,
                      rag_function) -> ComparisonResult:
        """Compara rendimiento entre OpenAI y modelo local"""
        
        # Medir OpenAI
        openai_metrics = self.measure_query_performance(
            query_text, "openai", rag_function, openai_function
        )
        
        # Medir modelo local
        local_metrics = self.measure_query_performance(
            query_text, "local", rag_function, local_function
        )
        
        # Crear comparación
        comparison = ComparisonResult(
            openai_metrics=openai_metrics,
            local_metrics=local_metrics,
            preference_score=None,  # Se puede evaluar manualmente
            evaluator_notes=None
        )
        
        self._save_comparison(comparison)
        return comparison
    
    def _save_comparison(self, comparison: ComparisonResult):
        """Guarda comparación en la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        comparison_id = hashlib.md5(
            f"{comparison.openai_metrics.query_id}_{comparison.local_metrics.query_id}".encode()
        ).hexdigest()[:12]
        
        cursor.execute('''
            INSERT OR REPLACE INTO comparisons VALUES (?,?,?,?,?,?,?)
        ''', (
            comparison_id,
            datetime.now().isoformat(),
            comparison.openai_metrics.query_text,
            comparison.openai_metrics.query_id,
            comparison.local_metrics.query_id,
            comparison.preference_score,
            comparison.evaluator_notes
        ))
        
        conn.commit()
        conn.close()
    
    def get_performance_summary(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Obtiene resumen de rendimiento"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        where_clause = "WHERE model_name = ?" if model_name else ""
        params = [model_name] if model_name else []
        
        cursor.execute(f'''
            SELECT 
                COUNT(*) as total_queries,
                AVG(latency_seconds) as avg_latency,
                MIN(latency_seconds) as min_latency,
                MAX(latency_seconds) as max_latency,
                AVG(fragments_retrieved) as avg_fragments,
                AVG(confidence_score) as avg_confidence,
                SUM(error_occurred) as total_errors
            FROM query_metrics {where_clause}
        ''', params)
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            "model_name": model_name or "all",
            "total_queries": result[0],
            "avg_latency": round(result[1], 3) if result[1] else 0,
            "min_latency": round(result[2], 3) if result[2] else 0,
            "max_latency": round(result[3], 3) if result[3] else 0,
            "avg_fragments": round(result[4], 1) if result[4] else 0,
            "avg_confidence": round(result[5], 3) if result[5] else 0,
            "error_rate": round(result[6] / result[0] * 100, 1) if result[0] > 0 else 0
        }
    
    def get_document_type_usage(self) -> Dict[str, int]:
        """Analiza qué tipos de documentos se usan más"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT document_types_used FROM query_metrics WHERE NOT error_occurred')
        results = cursor.fetchall()
        conn.close()
        
        type_counter = {}
        for row in results:
            types = json.loads(row[0]) if row[0] else []
            for doc_type in types:
                type_counter[doc_type] = type_counter.get(doc_type, 0) + 1
        
        return dict(sorted(type_counter.items(), key=lambda x: x[1], reverse=True))
    
    def export_metrics_for_tfm(self, output_file: str = "tfm_metrics_export.json"):
        """Exporta métricas en formato para TFM"""
        
        # Resúmenes por modelo
        openai_summary = self.get_performance_summary("openai")
        local_summary = self.get_performance_summary("local")
        overall_summary = self.get_performance_summary()
        
        # Uso por tipo de documento
        document_usage = self.get_document_type_usage()
        
        # Comparaciones
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM comparisons')
        total_comparisons = cursor.fetchone()[0]
        conn.close()
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "system_performance": {
                "openai": openai_summary,
                "local": local_summary,
                "overall": overall_summary
            },
            "document_type_analysis": document_usage,
            "comparison_statistics": {
                "total_comparisons": total_comparisons
            },
            "methodology_notes": {
                "rag_approach": "ChromaDB + LlamaIndex",
                "embedding_model": "all-MiniLM-L6-v2",
                "chunking_strategy": "specialized_by_document_type",
                "evaluation_criteria": [
                    "latency", "fragments_retrieved", "confidence_score", 
                    "error_rate", "document_type_coverage"
                ]
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return export_data

# Ejemplo de uso en el sistema
def create_metrics_evaluator():
    """Factory function para crear evaluador de métricas"""
    return MetricsEvaluator()

# Test cases predefinidos para TFM
TFM_TEST_QUERIES = [
    "¿Qué documentos necesito para solicitar una licencia de obras?",
    "¿Cuál es el procedimiento para presentar una reclamación municipal?",
    "¿Qué ordenanza regula el ruido en el municipio?",
    "¿Cómo consulto el estado de mi expediente electrónico?",
    "¿Cuáles son los horarios de atención al público?",
    "¿Qué subvenciones están disponibles para empresas locales?",
    "¿Cómo solicito el certificado de empadronamiento?",
    "¿Cuál es el proceso para presentar alegaciones a un expediente?",
    "¿Qué normativa regula las terrazas de hostelería?",
    "¿Cómo puedo acceder al portal de transparencia?"
]