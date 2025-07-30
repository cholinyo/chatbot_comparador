"""
LlamaIndex Ingestor - Especializado en documentos municipales
Optimizado para administraciones locales con detección automática de tipos documentales
"""

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    KeywordExtractor,
    TitleExtractor
)
from llama_index.core.node_parser import SimpleNodeParser
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import hashlib

# Configurar logging
logger = logging.getLogger(__name__)

class MunicipalDocumentIngestor:
    """
    Ingesta optimizada para documentos de administración local
    
    Características:
    - Detección automática de tipos documentales municipales
    - Chunking especializado por tipo de documento
    - Extracción de metadatos enriquecidos
    - Preservación de estructura jerárquica
    """
    
    # Patrones específicos para documentos municipales españoles
    MUNICIPAL_PATTERNS = {
        "ordenanza": {
            "keywords": ["ordenanza", "artículo", "título", "capítulo", "disposición", "régimen sancionador"],
            "structure_patterns": [r'artículo\s+\d+', r'título\s+[ivx]+', r'capítulo\s+[ivx]+'],
            "chunk_strategy": "legal_articles"
        },
        "acta": {
            "keywords": ["acta", "sesión", "punto del día", "acuerdo", "pleno", "orden del día"],
            "structure_patterns": [r'\d+[.\-\s]*punto', r'punto\s+\d+', r'acuerdo\s+\d+'],
            "chunk_strategy": "meeting_points"
        },
        "resolucion": {
            "keywords": ["resuelvo", "considerando", "por tanto", "resolución", "decreto", "visto"],
            "structure_patterns": [r'considerando\s+\w+', r'resuelvo\s+\w+', r'antecedente\s+\w+'],
            "chunk_strategy": "resolution_sections"
        },
        "presupuesto": {
            "keywords": ["partida", "euros", "gastos", "ingresos", "presupuesto", "ejercicio", "crédito"],
            "structure_patterns": [r'partida\s+\d+', r'capítulo\s+\d+', r'artículo\s+\d+'],
            "chunk_strategy": "budget_items"
        },
        "convenio": {
            "keywords": ["convenio", "colaboración", "acuerdo marco", "partes", "cláusulas"],
            "structure_patterns": [r'cláusula\s+\w+', r'anexo\s+[ivx]+'],
            "chunk_strategy": "contract_clauses"
        },
        "normativa": {
            "keywords": ["reglamento", "instrucción", "circular", "protocolo", "directriz"],
            "structure_patterns": [r'artículo\s+\d+', r'apartado\s+\d+'],
            "chunk_strategy": "regulatory_sections"
        },
        "padron": {
            "keywords": ["padrón", "censo", "habitantes", "población", "empadronamiento"],
            "structure_patterns": [r'distrito\s+\d+', r'sección\s+\d+'],
            "chunk_strategy": "demographic_sections"
        },
        "contrato": {
            "keywords": ["contrato", "licitación", "adjudicación", "concurso", "procedimiento"],
            "structure_patterns": [r'lote\s+\d+', r'anexo\s+[ivx]+'],
            "chunk_strategy": "contract_sections"
        }
    }
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        """
        Inicializar el ingestor con configuración optimizada para documentos legales
        
        Args:
            chunk_size: Tamaño de chunks (optimizado para documentos legales)
            chunk_overlap: Solapamiento entre chunks para preservar contexto
        """
        # Configurar splitter para documentos legales/administrativos
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n",
            secondary_chunking_regex=r"[.!?;][\s]*"  # Respetar puntuación legal
        )
        
        # Configurar node parser con extractores especializados
        self.node_parser = SimpleNodeParser.from_defaults(
            text_splitter=self.text_splitter,
            include_metadata=True,
            include_prev_next_rel=True
        )
        
        # Extractores de metadatos especializados
        self.extractors = [
            TitleExtractor(nodes=5),  # Extraer títulos de documentos
            KeywordExtractor(keywords=20),  # Palabras clave relevantes
        ]
        
        logger.info("MunicipalDocumentIngestor inicializado")
    
    def process_municipal_folder(self, folder_path: str, 
                               recursive: bool = True,
                               allowed_extensions: List[str] = None) -> List[Document]:
        """
        Procesa carpeta con documentos municipales
        
        Args:
            folder_path: Ruta a la carpeta de documentos
            recursive: Buscar en subcarpetas
            allowed_extensions: Extensiones permitidas
            
        Returns:
            Lista de documentos procesados con metadatos enriquecidos
        """
        if allowed_extensions is None:
            allowed_extensions = [".pdf", ".docx", ".txt", ".doc"]
        
        folder_path = Path(folder_path)
        if not folder_path.exists():
            logger.error(f"Carpeta no encontrada: {folder_path}")
            return []
        
        # Configurar lectores específicos
        file_extractor = {
            ".pdf": PDFReader(return_full_document=True),
            ".docx": DocxReader(),
            ".doc": DocxReader(),  # Intentar con DocxReader
            ".txt": None  # Usar lector por defecto
        }
        
        try:
            # Cargar documentos con SimpleDirectoryReader
            reader = SimpleDirectoryReader(
                input_dir=str(folder_path),
                file_extractor=file_extractor,
                recursive=recursive,
                exclude_hidden=True,
                filename_as_id=True,
                required_exts=allowed_extensions
            )
            
            documents = reader.load_data()
            logger.info(f"Cargados {len(documents)} documentos de {folder_path}")
            
            # Enriquecer metadatos por tipo de documento municipal
            enriched_docs = []
            for doc in documents:
                try:
                    enriched_doc = self._enrich_document_metadata(doc, folder_path)
                    enriched_docs.append(enriched_doc)
                except Exception as e:
                    logger.warning(f"Error procesando documento {doc.doc_id}: {e}")
                    # Añadir documento sin enriquecer si hay error
                    enriched_docs.append(doc)
            
            logger.info(f"Enriquecidos {len(enriched_docs)} documentos")
            return enriched_docs
            
        except Exception as e:
            logger.error(f"Error procesando carpeta {folder_path}: {e}")
            return []
    
    def _enrich_document_metadata(self, doc: Document, source_folder: Path) -> Document:
        """
        Enriquecer metadatos del documento con información municipal específica
        """
        # Detectar tipo de documento municipal
        doc_type, confidence = self._detect_municipal_document_type(doc.text)
        
        # Extraer información adicional
        file_info = self._extract_file_info(doc)
        text_stats = self._calculate_text_statistics(doc.text)
        
        # Crear hash único del contenido
        content_hash = hashlib.md5(doc.text.encode('utf-8')).hexdigest()
        
        # Enriquecer metadatos
        doc.metadata.update({
            # Clasificación documental
            "document_type": doc_type,
            "detection_confidence": confidence,
            
            # Información de archivo
            "source_folder": str(source_folder),
            "file_name": file_info.get("filename", "unknown"),
            "file_extension": file_info.get("extension", "unknown"),
            "file_size": file_info.get("size", 0),
            
            # Estadísticas de texto
            "text_length": text_stats["char_count"],
            "word_count": text_stats["word_count"],
            "paragraph_count": text_stats["paragraph_count"],
            
            # Metadatos de procesamiento
            "processing_timestamp": datetime.now().isoformat(),
            "content_hash": content_hash,
            "processed_by": "llamaindex_municipal",
            "chunk_strategy": self.MUNICIPAL_PATTERNS.get(doc_type, {}).get("chunk_strategy", "standard")
        })
        
        return doc
    
    def _detect_municipal_document_type(self, text: str) -> Tuple[str, float]:
        """
        Detecta tipo de documento municipal con score de confianza
        
        Returns:
            Tupla (tipo_documento, confianza)
        """
        text_lower = text.lower()
        scores = {}
        
        # Analizar cada tipo de documento
        for doc_type, pattern_info in self.MUNICIPAL_PATTERNS.items():
            score = 0
            
            # Puntuación por palabras clave
            keywords = pattern_info["keywords"]
            keyword_score = sum(1 for keyword in keywords if keyword in text_lower)
            
            # Puntuación por patrones estructurales
            structure_patterns = pattern_info["structure_patterns"]
            structure_score = 0
            for pattern in structure_patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                structure_score += matches
            
            # Calcular score total (ponderado)
            total_score = (keyword_score * 0.6) + (structure_score * 0.4)
            scores[doc_type] = total_score
        
        # Determinar el tipo con mayor score
        if not scores or max(scores.values()) == 0:
            return "documento_general", 0.0
        
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        
        # Calcular confianza (normalizada)
        total_keywords = sum(len(p["keywords"]) for p in self.MUNICIPAL_PATTERNS.values())
        confidence = min(max_score / (total_keywords * 0.1), 1.0)
        
        return best_type, confidence
    
    def _extract_file_info(self, doc: Document) -> Dict[str, Any]:
        """Extraer información del archivo"""
        file_info = {}
        
        # Información del archivo desde metadatos
        if hasattr(doc, 'metadata') and doc.metadata:
            file_path = doc.metadata.get('file_path', '')
            if file_path:
                path_obj = Path(file_path)
                file_info.update({
                    "filename": path_obj.name,
                    "extension": path_obj.suffix,
                    "size": path_obj.stat().st_size if path_obj.exists() else 0
                })
        
        return file_info
    
    def _calculate_text_statistics(self, text: str) -> Dict[str, int]:
        """Calcular estadísticas básicas del texto"""
        return {
            "char_count": len(text),
            "word_count": len(text.split()),
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()])
        }
    
    def create_specialized_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Crear chunks especializados según tipo de documento municipal
        
        Args:
            documents: Lista de documentos enriquecidos
            
        Returns:
            Lista de chunks especializados
        """
        specialized_chunks = []
        
        for doc in documents:
            doc_type = doc.metadata.get("document_type", "documento_general")
            chunk_strategy = doc.metadata.get("chunk_strategy", "standard")
            
            try:
                # Aplicar estrategia de chunking específica
                if chunk_strategy == "legal_articles":
                    chunks = self._chunk_legal_document(doc)
                elif chunk_strategy == "meeting_points":
                    chunks = self._chunk_meeting_document(doc)
                elif chunk_strategy == "resolution_sections":
                    chunks = self._chunk_resolution_document(doc)
                elif chunk_strategy == "budget_items":
                    chunks = self._chunk_budget_document(doc)
                elif chunk_strategy == "contract_clauses":
                    chunks = self._chunk_contract_document(