"""
Script de migración de FAISS a ChromaDB
Ejecutar desde la raíz del proyecto: python scripts/migrate_to_chroma.py
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Añadir el directorio raíz al path para imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.utils.chroma_store import get_chroma_store
from app.utils.rag_utils import (
    ingest_documents_with_llamaindex, 
    obtener_estadisticas_vectorstore,
    diagnosticar_vectorstore
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/migration.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def backup_existing_vectorstore():
    """Hacer backup del vectorstore existente"""
    try:
        backup_dir = f"backup/vectorstore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup de FAISS si existe
        faiss_dirs = ["vectorstore/documents", "vectorstore/web", "vectorstore/apis", "vectorstore/bbdd"]
        backup_count = 0
        
        for faiss_dir in faiss_dirs:
            if os.path.exists(faiss_dir):
                import shutil
                dest_dir = os.path.join(backup_dir, os.path.basename(faiss_dir))
                shutil.copytree(faiss_dir, dest_dir)
                backup_count += 1
                logger.info(f"📦 Backup realizado: {faiss_dir} → {dest_dir}")
        
        if backup_count > 0:
            logger.info(f"✅ Backup completado en: {backup_dir}")
            return backup_dir
        else:
            logger.info("ℹ️ No se encontraron datos existentes para backup")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error en backup: {e}")
        return None

def load_configuration():
    """Cargar configuración del sistema"""
    config_path = "app/config/settings.json"
    
    if not os.path.exists(config_path):
        logger.error(f"❌ No se encontró configuración en: {config_path}")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info("✅ Configuración cargada")
        return config
    except Exception as e:
        logger.error(f"❌ Error cargando configuración: {e}")
        return None

def migrate_documents(config):
    """Migrar documentos usando el nuevo sistema"""
    folders = config.get("document_folders", [])
    
    if not folders:
        logger.warning("⚠️ No hay carpetas de documentos configuradas")
        return 0
    
    logger.info(f"📁 Migrando {len(folders)} carpetas de documentos...")
    
    try:
        total_docs = ingest_documents_with_llamaindex(folders)
        logger.info(f"✅ Documentos migrados: {total_docs}")
        return total_docs
    except Exception as e:
        logger.error(f"❌ Error migrando documentos: {e}")
        return 0

def migrate_web_sources(config):
    """Migrar fuentes web - placeholder para implementación futura"""
    web_sources = config.get("web_sources", [])
    
    if not web_sources:
        logger.info("ℹ️ No hay fuentes web configuradas")
        return 0
    
    logger.warning(f"⚠️ Migración web pendiente: {len(web_sources)} fuentes")
    # TODO: Implementar migración de fuentes web
    return 0

def migrate_api_sources(config):
    """Migrar fuentes API - placeholder para implementación futura"""
    api_sources = config.get("api_sources", [])
    
    if not api_sources:
        logger.info("ℹ️ No hay fuentes API configuradas")
        return 0
    
    logger.warning(f"⚠️ Migración API pendiente: {len(api_sources)} fuentes")
    # TODO: Implementar migración de fuentes API
    return 0

def verify_migration():
    """Verificar que la migración fue exitosa"""
    logger.info("🔍 Verificando migración...")
    
    try:
        # Obtener estadísticas
        stats = obtener_estadisticas_vectorstore()
        
        if stats.get("total_documents", 0) > 0:
            logger.info(f"✅ Verificación exitosa:")
            logger.info(f"  - Total documentos: {stats['total_documents']}")
            logger.info(f"  - Backend: {stats['backend']}")
            logger.info(f"  - Campos metadata: {len(stats.get('metadata_fields', []))}")
            return True
        else:
            logger.error("❌ Verificación falló: No hay documentos en ChromaDB")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en verificación: {e}")
        return False

def test_search_functionality():
    """Probar funcionalidad de búsqueda"""
    logger.info("🧪 Probando funcionalidad de búsqueda...")
    
    try:
        from app.utils.rag_utils import buscar_fragmentos_combinados
        
        # Prueba básica
        test_queries = [
            "ordenanza municipal",
            "licencia obras",
            "presupuesto municipal",
            "acta sesión"
        ]
        
        for query in test_queries:
            results = buscar_fragmentos_combinados(query, k=3)
            logger.info(f"  Query '{query}': {len(results)} resultados")
        
        logger.info("✅ Pruebas de búsqueda completadas")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en pruebas de búsqueda: {e}")
        return False

def generate_migration_report(backup_dir, migration_stats):
    """Generar reporte de migración"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "backup_location": backup_dir,
        "migration_stats": migration_stats,
        "vectorstore_stats": obtener_estadisticas_vectorstore(),
        "diagnostic": diagnosticar_vectorstore()
    }
    
    # Guardar reporte
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Reporte generado: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"❌ Error generando reporte: {e}")
        return None

def main():
    """Función principal de migración"""
    logger.info("🚀 Iniciando migración FAISS → ChromaDB")
    
    # Crear directorios necesarios
    os.makedirs("logs", exist_ok=True)
    os.makedirs("backup", exist_ok=True)
    
    migration_stats = {
        "documents_migrated": 0,
        "web_sources_migrated": 0,
        "api_sources_migrated": 0,
        "success": False,
        "errors": []
    }
    
    try:
        # 1. Backup existente
        backup_dir = backup_existing_vectorstore()
        
        # 2. Cargar configuración
        config = load_configuration()
        if not config:
            migration_stats["errors"].append("No se pudo cargar la configuración")
            return False
        
        # 3. Migrar documentos
        docs_migrated = migrate_documents(config)
        migration_stats["documents_migrated"] = docs_migrated
        
        # 4. Migrar fuentes web (futuro)
        web_migrated = migrate_web_sources(config)
        migration_stats["web_sources_migrated"] = web_migrated
        
        # 5. Migrar APIs (futuro)
        api_migrated = migrate_api_sources(config)
        migration_stats["api_sources_migrated"] = api_migrated
        
        # 6. Verificar migración
        if verify_migration():
            # 7. Probar búsqueda
            if test_search_functionality():
                migration_stats["success"] = True
                logger.info("🎉 Migración completada exitosamente")
            else:
                migration_stats["errors"].append("Falló prueba de búsqueda")
        else:
            migration_stats["errors"].append("Falló verificación de datos")
        
        # 8. Generar reporte
        report_path = generate_migration_report(backup_dir, migration_stats)
        
        return migration_stats["success"]
        
    except Exception as e:
        logger.error(f"❌ Error crítico en migración: {e}")
        migration_stats["errors"].append(str(e))
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*60)
        print("🎉 MIGRACIÓN EXITOSA")
        print("="*60)
        print("✅ ChromaDB está listo para usar")
        print("✅ Sistema RAG actualizado")
        print("✅ Búsquedas funcionando")
        print("\n📝 Próximos pasos:")
        print("   1. Actualizar imports en el código")
        print("   2. Probar chat y comparador")
        print("   3. Migrar fuentes web/API si es necesario")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ MIGRACIÓN FALLÓ")
        print("="*60)
        print("💡 Revisar logs en: logs/migration.log")
        print("💡 Restaurar backup si es necesario")
        print("="*60)
    
    sys.exit(0 if success else 1)