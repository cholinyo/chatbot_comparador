"""
RAG Utils - Actualizado con ChromaDB + LlamaIndex
Mantiene compatibilidad con el sistema existente
"""
import os
import logging
from typing import List, Dict, Any, Optional
from app.utils.chroma_store import get_chroma_store
from app.services.llamaindex_ingestor import MunicipalDocumentIngestor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FUNCIONES PRINCIPALES RAG
# ============================================================================

def buscar_fragmentos_combinados(
    consulta: str, 
    k: int = 5, 
    filtros: Optional[Dict] = None,
    fuente_especifica: Optional[str] = None
) -> List[Dict]:
    """
    Búsqueda mejorada con ChromaDB y filtros avanzados
    
    Args:
        consulta: Texto de la consulta
        k: Número de fragmentos a recuperar
        filtros: Filtros de metadatos (ej: {"document_type": "ordenanza"})
        fuente_especifica: Filtrar por fuente específica
    
    Returns:
        Lista de fragmentos con metadatos enriquecidos
    """
    try:
        store = get_chroma_store()
        
        # Construir filtros
        search_filters = {}
        if filtros:
            search_filters.update(filtros)
        if fuente_especifica:
            search_filters["fuente"] = fuente_especifica
        
        # Realizar búsqueda
        results = store.similarity_search(
            query=consulta,
            k=k,
            filter_metadata=search_filters if search_filters else None
        )
        
        logger.info(f"🔍 Búsqueda '{consulta[:50]}...': {len(results)} fragmentos encontrados")
        
        # Enriquecer resultados con información adicional
        fragmentos_enriquecidos = []
        for i, fragmento in enumerate(results):
            fragmento_enriquecido = {
                **fragmento,
                "ranking": i + 1,
                "relevancia_score": round(1.0 - (i * 0.1), 2),  # Score simulado
                "fragmento_id": fragmento.get("metadata", {}).get("id", f"frag_{i}"),
                "tipo_documento": fragmento.get("metadata", {}).get("document_type", "general")
            }
            fragmentos_enriquecidos.append(fragmento_enriquecido)
        
        return fragmentos_enriquecidos
        
    except Exception as e:
        logger.error(f"❌ Error en búsqueda combinada: {e}")
        return []

def ingest_documents_with_llamaindex(folder_paths: List[str]) -> int:
    """
    Ingesta documentos usando LlamaIndex + ChromaDB
    
    Args:
        folder_paths: Lista de rutas de carpetas a procesar
    
    Returns:
        Número total de documentos procesados
    """
    if not folder_paths:
        logger.warning("⚠️ No se proporcionaron carpetas para ingestar")
        return 0
    
    ingestor = MunicipalDocumentIngestor()
    store = get_chroma_store()
    
    total_docs = 0
    
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            logger.warning(f"⚠️ Carpeta no encontrada: {folder_path}")
            continue
            
        try:
            logger.info(f"📁 Procesando carpeta: {folder_path}")
            
            # Procesar con LlamaIndex
            docs = ingestor.process_municipal_folder(folder_path)
            
            if not docs:
                logger.warning(f"⚠️ No se encontraron documentos en: {folder_path}")
                continue
            
            # Crear chunks especializados
            chunks = ingestor.create_specialized_chunks(docs)
            
            if chunks:
                # Preparar textos y metadatos para ChromaDB
                texts = []
                metadatas = []
                
                for chunk in chunks:
                    texts.append(chunk.text)
                    
                    # Enriquecer metadatos con información de la carpeta
                    metadata = chunk.metadata.copy()
                    metadata.update({
                        "fuente": "documentos",
                        "carpeta_origen": folder_path,
                        "metodo_ingesta": "llamaindex_v2"
                    })
                    metadatas.append(metadata)
                
                # Añadir a ChromaDB
                ids = store.add_documents(texts, metadatas)
                docs_added = len(ids)
                total_docs += docs_added
                
                logger.info(f"✅ Carpeta {folder_path}: {docs_added} fragmentos ingestados")
            
        except Exception as e:
            logger.error(f"❌ Error procesando carpeta {folder_path}: {e}")
            continue
    
    logger.info(f"🎯 Ingesta completada: {total_docs} fragmentos totales")
    return total_docs

def get_vectorstore_stats() -> Dict[str, Any]:
    """Obtener estadísticas del vectorstore actual"""
    try:
        store = get_chroma_store()
        stats = store.get_collection_stats()
        
        # Añadir información adicional
        stats.update({
            "vectorstore_type": "ChromaDB",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "status": "activo"
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas: {e}")
        return {"error": str(e), "status": "error"}

# ============================================================================
# FUNCIONES DE COMPATIBILIDAD (mantener interfaz existente)
# ============================================================================

def buscar_fragmentos(consulta: str, k: int = 3) -> List[Dict]:
    """
    Función de compatibilidad con código existente
    Mapea a la nueva función buscar_fragmentos_combinados
    """
    return buscar_fragmentos_combinados(consulta, k)

def buscar_fragmentos_por_fuente(
    consulta: str, 
    fuente: str, 
    k: int = 5
) -> List[Dict]:
    """
    Búsqueda filtrada por fuente específica
    
    Args:
        consulta: Texto de consulta
        fuente: Fuente específica (documentos, web, apis, bbdd)
        k: Número de resultados
    """
    filtros = {"fuente": fuente}
    return buscar_fragmentos_combinados(consulta, k, filtros)

def buscar_por_tipo_documento(
    consulta: str, 
    tipo_documento: str, 
    k: int = 5
) -> List[Dict]:
    """
    Búsqueda filtrada por tipo de documento municipal
    
    Args:
        consulta: Texto de consulta
        tipo_documento: Tipo (ordenanza, acta, resolucion, etc.)
        k: Número de resultados
    """
    filtros = {"document_type": tipo_documento}
    return buscar_fragmentos_combinados(consulta, k, filtros)

# ============================================================================
# FUNCIONES DE MIGRACIÓN Y MANTENIMIENTO
# ============================================================================

def migrate_from_faiss_to_chroma(config_path: str = "app/config/settings.json") -> Dict[str, Any]:
    """
    Migra datos existentes de FAISS a ChromaDB
    
    Args:
        config_path: Ruta al archivo de configuración
    
    Returns:
        Reporte de migración
    """
    import json
    
    try:
        logger.info("🔄 Iniciando migración FAISS -> ChromaDB...")
        
        # Cargar configuración
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Re-ingestar documentos con nuevo sistema
        folders = config.get("document_folders", [])
        
        if not folders:
            return {"error": "No hay carpetas configuradas", "migrated_docs": 0}
        
        # Limpiar colección existente
        store = get_chroma_store()
        try:
            store.delete_collection()
            store = get_chroma_store()  # Recrear
        except:
            pass  # La colección puede no existir
        
        # Ingestar con nuevo sistema
        total_docs = ingest_documents_with_llamaindex(folders)
        
        # Generar reporte
        stats = get_vectorstore_stats()
        
        migration_report = {
            "status": "success",
            "migrated_docs": total_docs,
            "vectorstore_stats": stats,
            "migration_timestamp": logger.info.__module__  # Placeholder
        }
        
        logger.info(f"✅ Migración completada: {total_docs} documentos")
        return migration_report
        
    except Exception as e:
        logger.error(f"❌ Error en migración: {e}")
        return {"status": "error", "error": str(e), "migrated_docs": 0}

def cleanup_vectorstore():
    """Limpiar completamente el vectorstore"""
    try:
        store = get_chroma_store()
        store.delete_collection()
        logger.info("🗑️ Vectorstore limpiado completamente")
        return True
    except Exception as e:
        logger.error(f"❌ Error limpiando vectorstore: {e}")
        return False

def reindex_single_folder(folder_path: str) -> int:
    """
    Reindexar una sola carpeta
    
    Args:
        folder_path: Ruta de la carpeta a reindexar
    
    Returns:
        Número de documentos procesados
    """
    return ingest_documents_with_llamaindex([folder_path])

# ============================================================================
# FUNCIONES DE ANÁLISIS Y DEBUGGING
# ============================================================================

def analyze_document_types() -> Dict[str, Any]:
    """Analizar distribución de tipos de documentos en el vectorstore"""
    try:
        store = get_chroma_store()
        
        # Buscar todos los documentos por tipo
        types_analysis = {}
        document_types = [
            "ordenanza", "acta", "resolucion", "presupuesto", 
            "convenio", "normativa", "subvencion", "licencia", "documento_general"
        ]
        
        for doc_type in document_types:
            results = store.search_by_metadata(
                {"document_type": doc_type}, 
                limit=1000
            )
            types_analysis[doc_type] = len(results)
        
        # Estadísticas generales
        total_docs = sum(types_analysis.values())
        
        analysis_report = {
            "total_documents": total_docs,
            "document_types": types_analysis,
            "most_common_type": max(types_analysis, key=types_analysis.get) if total_docs > 0 else None,
            "type_diversity": len([t for t, count in types_analysis.items() if count > 0])
        }
        
        return analysis_report
        
    except Exception as e:
        logger.error(f"❌ Error en análisis de tipos: {e}")
        return {"error": str(e)}

def test_search_performance(test_queries: List[str], k: int = 5) -> Dict[str, Any]:
    """
    Probar rendimiento de búsqueda con consultas de test
    
    Args:
        test_queries: Lista de consultas de prueba
        k: Número de resultados por consulta
    
    Returns:
        Reporte de rendimiento
    """
    import time
    
    performance_data = {
        "queries_tested": len(test_queries),
        "average_response_time": 0,
        "queries_results": []
    }
    
    total_time = 0
    
    for query in test_queries:
        start_time = time.time()
        
        try:
            results = buscar_fragmentos_combinados(query, k)
            response_time = time.time() - start_time
            
            performance_data["queries_results"].append({
                "query": query,
                "response_time_ms": round(response_time * 1000, 2),
                "results_count": len(results),
                "success": True
            })
            
            total_time += response_time
            
        except Exception as e:
            performance_data["queries_results"].append({
                "query": query,
                "error": str(e),
                "success": False
            })
    
    if len(test_queries) > 0:
        performance_data["average_response_time"] = round(total_time / len(test_queries) * 1000, 2)
    
    return performance_data

# ============================================================================
# CONFIGURACIÓN Y CONSTANTES
# ============================================================================

# Consultas de test para evaluación
DEFAULT_TEST_QUERIES = [
    "¿Qué documentos necesito para una licencia de obras?",
    "Ordenanza municipal de ruidos",
    "Presupuesto municipal 2024",
    "Convocatoria de subvenciones",
    "Actas del último pleno",
    "Reglamento de participación ciudadana"
]

# Tipos de documentos soportados
SUPPORTED_DOCUMENT_TYPES = [
    "ordenanza", "acta", "resolucion", "presupuesto", 
    "convenio", "normativa", "subvencion", "licencia", "documento_general"
]

# Configuración por defecto
DEFAULT_RAG_CONFIG = {
    "chunk_size": 1024,
    "chunk_overlap": 200,
    "max_results": 10,
    "embedding_model": "all-MiniLM-L6-v2"
}