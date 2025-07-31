"""
Script de validación completa del sistema ChromaDB + LlamaIndex
Ejecutar después de la migración para verificar funcionalidades
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.chroma_store import get_chroma_store
from app.utils.rag_utils import (
    buscar_fragmentos_combinados,
    buscar_por_tipo_documento,
    obtener_tipos_documento_disponibles,
    obtener_estadisticas_vectorstore
)
from app.services.llamaindex_ingestor import MunicipalDocumentIngestor
import json
from datetime import datetime

def test_chromadb_connection():
    """Test básico de conexión a ChromaDB"""
    print("🔍 Verificando conexión ChromaDB...")
    try:
        store = get_chroma_store()
        stats = store.get_collection_stats()
        print(f"✅ ChromaDB conectado: {stats['total_documents']} documentos")
        return True, stats
    except Exception as e:
        print(f"❌ Error conectando ChromaDB: {e}")
        return False, None

def test_document_types_detection():
    """Test detección automática de tipos documentales"""
    print("\n📋 Verificando detección de tipos documentales...")
    try:
        tipos = obtener_tipos_documento_disponibles()
        print(f"✅ Tipos detectados: {len(tipos)}")
        for tipo in tipos:
            print(f"   - {tipo}")
        return True, tipos
    except Exception as e:
        print(f"❌ Error en detección tipos: {e}")
        return False, []

def test_semantic_search():
    """Test búsqueda semántica básica"""
    print("\n🔎 Verificando búsqueda semántica...")
    test_queries = [
        "licencia de obras",
        "ordenanza municipal",
        "punto del día",
        "presupuesto municipal"
    ]
    
    results = {}
    for query in test_queries:
        try:
            fragmentos = buscar_fragmentos_combinados(query, k=3)
            results[query] = len(fragmentos)
            print(f"✅ '{query}': {len(fragmentos)} fragmentos encontrados")
        except Exception as e:
            results[query] = f"Error: {e}"
            print(f"❌ '{query}': {e}")
    
    return results

def test_filtered_search():
    """Test búsqueda con filtros por tipo"""
    print("\n🎯 Verificando búsqueda filtrada por tipo...")
    try:
        # Buscar ordenanzas
        ordenanzas = buscar_por_tipo_documento("licencia", "ordenanza", k=2)
        print(f"✅ Búsqueda en ordenanzas: {len(ordenanzas)} resultados")
        
        # Buscar actas
        actas = buscar_por_tipo_documento("punto", "acta", k=2)
        print(f"✅ Búsqueda en actas: {len(actas)} resultados")
        
        return True
    except Exception as e:
        print(f"❌ Error en búsqueda filtrada: {e}")
        return False

def test_specialized_chunking():
    """Test chunking especializado"""
    print("\n✂️ Verificando chunking especializado...")
    try:
        ingestor = MunicipalDocumentIngestor()
        
        # Test texto ordenanza
        test_ordenanza = """
        ORDENANZA MUNICIPAL DE RUIDOS
        
        Artículo 1. Objeto y ámbito de aplicación.
        La presente ordenanza tiene por objeto...
        
        Artículo 2. Definiciones.
        A efectos de esta ordenanza se entiende por...
        """
        
        tipo_detectado = ingestor._detect_municipal_document_type(test_ordenanza)
        print(f"✅ Tipo detectado para texto test: '{tipo_detectado}'")
        
        return True, tipo_detectado
    except Exception as e:
        print(f"❌ Error en chunking especializado: {e}")
        return False, None

def generate_validation_report():
    """Genera reporte completo de validación"""
    print("\n📊 Generando reporte de validación...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "validation_results": {}
    }
    
    # Test 1: Conexión ChromaDB
    chromadb_ok, stats = test_chromadb_connection()
    report["validation_results"]["chromadb_connection"] = {
        "status": "OK" if chromadb_ok else "ERROR",
        "stats": stats
    }
    
    # Test 2: Detección tipos
    types_ok, tipos = test_document_types_detection()
    report["validation_results"]["document_types"] = {
        "status": "OK" if types_ok else "ERROR",
        "types_detected": tipos,
        "count": len(tipos) if types_ok else 0
    }
    
    # Test 3: Búsqueda semántica
    search_results = test_semantic_search()
    report["validation_results"]["semantic_search"] = {
        "status": "OK" if all(isinstance(v, int) for v in search_results.values()) else "PARTIAL",
        "queries_tested": search_results
    }
    
    # Test 4: Búsqueda filtrada
    filtered_ok = test_filtered_search()
    report["validation_results"]["filtered_search"] = {
        "status": "OK" if filtered_ok else "ERROR"
    }
    
    # Test 5: Chunking especializado
    chunking_ok, tipo_test = test_specialized_chunking()
    report["validation_results"]["specialized_chunking"] = {
        "status": "OK" if chunking_ok else "ERROR",
        "test_detection": tipo_test
    }
    
    # Guardar reporte
    report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Reporte guardado en: {report_file}")
    return report

def main():
    """Función principal de validación"""
    print("🚀 VALIDACIÓN SISTEMA CHROMADB + LLAMAINDEX")
    print("=" * 50)
    
    # Ejecutar todas las validaciones
    report = generate_validation_report()
    
    # Resumen final
    print("\n📋 RESUMEN DE VALIDACIÓN:")
    print("=" * 30)
    
    all_ok = True
    for test_name, result in report["validation_results"].items():
        status = result["status"]
        icon = "✅" if status == "OK" else "⚠️" if status == "PARTIAL" else "❌"
        print(f"{icon} {test_name.replace('_', ' ').title()}: {status}")
        if status != "OK":
            all_ok = False
    
    if all_ok:
        print("\n🎉 ¡SISTEMA COMPLETAMENTE VALIDADO!")
        print("✅ Listo para continuar con desarrollo avanzado")
    else:
        print("\n⚠️ Algunas validaciones requieren atención")
        print("🔧 Revisa los errores antes de continuar")
    
    return all_ok

if __name__ == "__main__":
    main()