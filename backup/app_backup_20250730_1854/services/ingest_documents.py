"""
Mejoras propuestas para la ingesta de documentos en TFM
Siguiendo principios de gestión documental digital y optimización
"""

import os
import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json

# 1. INGESTA ASÍNCRONA Y PARALELA
class AsyncDocumentIngestor:
    """Mejorado: Procesamiento asíncrono de documentos"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.processed_hashes = set()  # Evitar duplicados
        
    async def process_documents_batch(self, document_paths: List[str]) -> List[Dict]:
        """Procesa documentos en lotes de forma asíncrona"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, self.process_single_document, path)
                for path in document_paths
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        # Filtrar errores y resultados válidos
        return [r for r in results if isinstance(r, dict)]
    
    def process_single_document(self, path: str) -> Dict[str, Any]:
        """Procesa un documento individual con metadatos enriquecidos"""
        try:
            # Calcular hash para evitar duplicados
            file_hash = self._calculate_file_hash(path)
            if file_hash in self.processed_hashes:
                return {"status": "duplicated", "path": path}
            
            self.processed_hashes.add(file_hash)
            
            # Metadatos enriquecidos
            metadata = self._extract_enhanced_metadata(path)
            
            # Procesamiento según tipo
            if path.endswith('.pdf'):
                content = self._process_pdf_advanced(path)
            elif path.endswith('.docx'):
                content = self._process_docx_advanced(path)
            else:
                content = self._process_text_file(path)
            
            return {
                "path": path,
                "content": content,
                "metadata": metadata,
                "hash": file_hash,
                "status": "processed"
            }
            
        except Exception as e:
            return {"status": "error", "path": path, "error": str(e)}


# 2. DETECCIÓN INTELIGENTE DE ESTRUCTURA
class SmartDocumentStructure:
    """Detecta automáticamente la estructura del documento"""
    
    def detect_document_type(self, text: str) -> str:
        """Detecta tipo de documento por contenido"""
        keywords = {
            'ordenanza': ['ordenanza', 'artículo', 'capítulo', 'disposición'],
            'acta': ['acta', 'sesión', 'punto del día', 'acuerdo'],
            'resolución': ['resuelvo', 'considerando', 'por tanto'],
            'manual': ['procedimiento', 'paso', 'instrucciones'],
            'presupuesto': ['partida', 'euros', 'gastos', 'ingresos']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for doc_type, terms in keywords.items():
            score = sum(1 for term in terms if term in text_lower)
            scores[doc_type] = score
            
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'generico'
    
    def extract_sections(self, text: str, doc_type: str) -> List[Dict]:
        """Extrae secciones según el tipo de documento"""
        if doc_type == 'ordenanza':
            return self._extract_legal_sections(text)
        elif doc_type == 'acta':
            return self._extract_meeting_sections(text)
        else:
            return self._extract_generic_sections(text)


# 3. FILTROS INTELIGENTES Y LIMPIEZA
class AdvancedTextCleaner:
    """Limpieza inteligente de texto"""
    
    def clean_document(self, text: str, doc_type: str) -> str:
        """Limpieza específica según tipo de documento"""
        # Limpieza general
        text = self._remove_headers_footers(text)
        text = self._fix_encoding_issues(text)
        text = self._normalize_whitespace(text)
        
        # Limpieza específica por tipo
        if doc_type == 'legal':
            text = self._clean_legal_document(text)
        elif doc_type == 'technical':
            text = self._clean_technical_document(text)
            
        return text
    
    def extract_key_entities(self, text: str) -> Dict[str, List[str]]:
        """Extrae entidades clave usando NLP básico"""
        import re
        
        entities = {
            'fechas': re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', text),
            'importes': re.findall(r'\d+[.,]\d+\s*€|\d+\s*euros?', text, re.IGNORECASE),
            'referencias': re.findall(r'art[íi]culo\s+\d+|ley\s+\d+/\d+', text, re.IGNORECASE),
            'organismos': self._extract_organizations(text)
        }
        
        return entities


# 4. MONITOREO Y MÉTRICAS
class IngestionMonitor:
    """Sistema de monitoreo de la ingesta"""
    
    def __init__(self):
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'errors': 0,
            'duplicates': 0,
            'processing_time': 0,
            'file_types': {},
            'error_log': []
        }
    
    def log_processing(self, file_path: str, status: str, processing_time: float):
        """Registra el procesamiento de un archivo"""
        self.stats['total_files'] += 1
        
        if status == 'processed':
            self.stats['processed_files'] += 1
        elif status == 'error':
            self.stats['errors'] += 1
        elif status == 'duplicated':
            self.stats['duplicates'] += 1
            
        self.stats['processing_time'] += processing_time
        
        # Estadísticas por tipo de archivo
        file_ext = Path(file_path).suffix.lower()
        self.stats['file_types'][file_ext] = self.stats['file_types'].get(file_ext, 0) + 1
    
    def generate_report(self) -> Dict[str, Any]:
        """Genera reporte de la ingesta"""
        return {
            'summary': self.stats,
            'success_rate': (self.stats['processed_files'] / self.stats['total_files']) * 100 if self.stats['total_files'] > 0 else 0,
            'avg_processing_time': self.stats['processing_time'] / self.stats['total_files'] if self.stats['total_files'] > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }


# 5. INTEGRACIÓN CON LANGCHAIN MEJORADA
class LangChainDocumentProcessor:
    """Integración optimizada con LangChain"""
    
    def __init__(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.document_loaders import DirectoryLoader
        
        # Configuración optimizada para administración local
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Tamaño optimizado
            chunk_overlap=200,  # Solapamiento para contexto
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def create_documents_with_metadata(self, processed_docs: List[Dict]) -> List:
        """Crea documentos LangChain con metadatos enriquecidos"""
        from langchain.schema import Document
        
        documents = []
        for doc in processed_docs:
            if doc['status'] == 'processed':
                # Dividir en chunks
                chunks = self.text_splitter.split_text(doc['content'])
                
                for i, chunk in enumerate(chunks):
                    metadata = {
                        **doc['metadata'],
                        'chunk_id': i,
                        'total_chunks': len(chunks),
                        'file_hash': doc['hash'],
                        'source_file': doc['path']
                    }
                    
                    documents.append(Document(
                        page_content=chunk,
                        metadata=metadata
                    ))
        
        return documents


# 6. CONFIGURACIÓN DE USO
async def main_improved_ingestion():
    """Función principal de ingesta mejorada"""
    
    # 1. Configurar componentes
    ingestor = AsyncDocumentIngestor(max_workers=4)
    monitor = IngestionMonitor()
    cleaner = AdvancedTextCleaner()
    structure_detector = SmartDocumentStructure()
    langchain_processor = LangChainDocumentProcessor()
    
    # 2. Obtener archivos a procesar
    config_path = "app/config/settings.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    document_folders = config.get('document_folders', [])
    all_files = []
    
    for folder in document_folders:
        if os.path.exists(folder):
            for ext in ['*.pdf', '*.docx', '*.txt']:
                all_files.extend(Path(folder).rglob(ext))
    
    # 3. Procesar en lotes
    batch_size = 10
    all_processed = []
    
    for i in range(0, len(all_files), batch_size):
        batch = [str(f) for f in all_files[i:i+batch_size]]
        processed_batch = await ingestor.process_documents_batch(batch)
        all_processed.extend(processed_batch)
        
        print(f"Procesado lote {i//batch_size + 1}/{len(all_files)//batch_size + 1}")
    
    # 4. Crear documentos LangChain
    documents = langchain_processor.create_documents_with_metadata(all_processed)
    
    # 5. Guardar en vectorstore
    # ... código de vectorización ...
    
    # 6. Generar reporte
    report = monitor.generate_report()
    print(f"Ingesta completada: {report}")
    
    return documents, report


# EJECUTAR MEJORA
if __name__ == "__main__":
    # Para ejecutar la ingesta mejorada
    documents, report = asyncio.run(main_improved_ingestion())
    print(f"Documentos procesados: {len(documents)}")
    print(f"Reporte: {report}")