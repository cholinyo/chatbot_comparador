"""
LlamaIndex Ingestor - Especializado en documentos municipales
Detección inteligente de tipos documentales y chunking optimizado
"""
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# LlamaIndex imports
try:
    from llama_index.core import SimpleDirectoryReader, Document
    from llama_index.readers.file import PDFReader, DocxReader
    from llama_index.core.text_splitter import SentenceSplitter
    from llama_index.core.node_parser import SimpleNodeParser
except ImportError:
    # Fallback para versiones anteriores
    from llama_index import SimpleDirectoryReader, Document
    from llama_index.readers.file import PDFReader, DocxReader
    from llama_index.text_splitter import SentenceSplitter
    from llama_index.node_parser import SimpleNodeParser

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MunicipalDocumentIngestor:
    """Ingesta optimizada para documentos de administración local"""
    
    def __init__(self):
        # Configurar splitter para documentos legales y administrativos
        self.text_splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200,
            separator=" ",
            paragraph_separator="\n\n",
            secondary_chunking_regex=r"[.!?]"
        )
        
        # Parser de nodos
        self.node_parser = SimpleNodeParser.from_defaults(
            include_metadata=True,
            include_prev_next_rel=True
        )
        
        logger.info("✅ MunicipalDocumentIngestor inicializado")
    
    def process_municipal_folder(self, folder_path: str) -> List[Document]:
        """
        Procesa carpeta con documentos municipales
        
        Args:
            folder_path: Ruta a la carpeta con documentos
            
        Returns:
            Lista de documentos procesados con metadatos enriquecidos
        """
        if not os.path.exists(folder_path):
            logger.warning(f"⚠️ Carpeta no encontrada: {folder_path}")
            return []
        
        # Configurar lectores específicos por tipo de archivo
        file_extractor = {
            ".pdf": PDFReader(return_full_document=True),
            ".docx": DocxReader(),
            ".txt": None  # Usar lector por defecto
        }
        
        logger.info(f"📁 Procesando carpeta: {folder_path}")
        
        try:
            # Cargar documentos
            reader = SimpleDirectoryReader(
                input_dir=folder_path,
                file_extractor=file_extractor,
                recursive=True,
                exclude_hidden=True,
                filename_as_id=True
            )
            
            documents = reader.load_data()
            logger.info(f"📄 Cargados {len(documents)} documentos")
            
            # Enriquecer metadatos por tipo de documento
            enriched_docs = []
            for doc in documents:
                try:
                    # Detectar tipo de documento municipal
                    doc_type = self._detect_municipal_document_type(doc.text)
                    
                    # Obtener información del archivo
                    file_info = self._extract_file_info(doc.metadata.get('file_name', ''))
                    
                    # Enriquecer metadatos
                    doc.metadata.update({
                        "document_type": doc_type,
                        "processed_by": "llamaindex",
                        "chunk_strategy": "sentence_based",
                        "source_folder": folder_path,
                        "processed_at": datetime.now().isoformat(),
                        "file_extension": file_info["extension"],
                        "file_size_kb": file_info["size_kb"],
                        "confidence_score": self._calculate_confidence_score(doc.text, doc_type)
                    })
                    
                    enriched_docs.append(doc)
                    
                except Exception as e:
                    logger.error(f"❌ Error procesando documento {doc.metadata.get('file_name', 'unknown')}: {e}")
                    continue
            
            logger.info(f"✅ Enriquecidos {len(enriched_docs)} documentos")
            return enriched_docs
            
        except Exception as e:
            logger.error(f"❌ Error procesando carpeta {folder_path}: {e}")
            return []
    
    def _detect_municipal_document_type(self, text: str) -> str:
        """
        Detecta tipo de documento municipal usando patrones específicos
        
        Args:
            text: Contenido del documento
            
        Returns:
            Tipo de documento detectado
        """
        if not text:
            return "documento_vacio"
            
        text_lower = text.lower()
        
        # Patrones específicos de administración local española
        patterns = {
            "ordenanza": {
                "keywords": ["ordenanza", "artículo", "título", "capítulo", "disposición", "boe", "bop"],
                "regex": [r"ordenanza\s+municipal", r"artículo\s+\d+", r"título\s+[ivx]+"]
            },
            "acta": {
                "keywords": ["acta", "sesión", "punto del día", "acuerdo", "pleno", "comisión"],
                "regex": [r"acta\s+de\s+la\s+sesión", r"punto\s+\d+", r"acuerdo\s+número"]
            },
            "resolucion": {
                "keywords": ["resuelvo", "considerando", "por tanto", "resolución", "alcaldía"],
                "regex": [r"resolución\s+número", r"considerando\s+que", r"por\s+tanto"]
            },
            "presupuesto": {
                "keywords": ["partida", "euros", "gastos", "ingresos", "presupuesto", "capítulo"],
                "regex": [r"partida\s+\d+", r"\d+[.,]\d+\s*€", r"capítulo\s+[ivx]+"]
            },
            "convenio": {
                "keywords": ["convenio", "colaboración", "acuerdo marco", "partes"],
                "regex": [r"convenio\s+de\s+colaboración", r"acuerdo\s+marco"]
            },
            "normativa": {
                "keywords": ["reglamento", "instrucción", "circular", "protocolo", "normativa"],
                "regex": [r"reglamento\s+municipal", r"instrucción\s+técnica"]
            },
            "subvencion": {
                "keywords": ["subvención", "ayuda", "beca", "convocatoria", "bases"],
                "regex": [r"convocatoria\s+de\s+subvenciones", r"bases\s+reguladoras"]
            },
            "licencia": {
                "keywords": ["licencia", "autorización", "permiso", "actividad", "obras"],
                "regex": [r"licencia\s+de\s+obras", r"licencia\s+de\s+actividad"]
            }
        }
        
        # Calcular puntuaciones
        scores = {}
        for doc_type, pattern_info in patterns.items():
            score = 0
            
            # Puntuación por keywords
            for keyword in pattern_info["keywords"]:
                score += text_lower.count(keyword)
            
            # Puntuación por regex (peso mayor)
            for regex_pattern in pattern_info["regex"]:
                matches = len(re.findall(regex_pattern, text_lower))
                score += matches * 3  # Peso mayor para patrones regex
            
            scores[doc_type] = score
        
        # Determinar tipo con mayor puntuación
        if max(scores.values()) > 0:
            detected_type = max(scores, key=scores.get)
            logger.debug(f"🔍 Documento detectado como: {detected_type} (score: {scores[detected_type]})")
            return detected_type
        
        return "documento_general"
    
    def _extract_file_info(self, filename: str) -> Dict[str, Any]:
        """Extrae información del archivo"""
        try:
            if not filename or not os.path.exists(filename):
                return {"extension": "unknown", "size_kb": 0}
                
            path = Path(filename)
            size_bytes = path.stat().st_size
            
            return {
                "extension": path.suffix.lower(),
                "size_kb": round(size_bytes / 1024, 2)
            }
        except:
            return {"extension": "unknown", "size_kb": 0}
    
    def _calculate_confidence_score(self, text: str, doc_type: str) -> float:
        """Calcula puntuación de confianza para la clasificación"""
        if not text or doc_type == "documento_general":
            return 0.5
        
        # Factores de confianza
        text_length = len(text)
        if text_length < 100:
            return 0.3  # Texto muy corto
        elif text_length > 5000:
            return 0.9  # Texto extenso, mayor confianza
        else:
            return 0.7  # Texto moderado
    
    def create_specialized_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Crea chunks especializados según tipo de documento
        
        Args:
            documents: Lista de documentos a chunkar
            
        Returns:
            Lista de chunks especializados
        """
        specialized_chunks = []
        
        for doc in documents:
            doc_type = doc.metadata.get("document_type", "general")
            
            try:
                if doc_type == "ordenanza":
                    chunks = self._chunk_legal_document(doc)
                elif doc_type == "acta":
                    chunks = self._chunk_meeting_document(doc)
                elif doc_type == "presupuesto":
                    chunks = self._chunk_budget_document(doc)
                else:
                    # Chunking estándar con nodos
                    nodes = self.node_parser.get_nodes_from_documents([doc])
                    chunks = [Document(text=node.text, metadata=doc.metadata) for node in nodes]
                
                specialized_chunks.extend(chunks)
                logger.debug(f"📝 {doc_type}: {len(chunks)} chunks creados")
                
            except Exception as e:
                logger.error(f"❌ Error chunking documento {doc_type}: {e}")
                # Fallback a chunking estándar
                chunks = self.text_splitter.split_text(doc.text)
                chunks = [Document(text=chunk, metadata=doc.metadata) for chunk in chunks]
                specialized_chunks.extend(chunks)
        
        logger.info(f"✅ Total chunks especializados: {len(specialized_chunks)}")
        return specialized_chunks
    
    def _chunk_legal_document(self, doc: Document) -> List[Document]:
        """Chunking especializado para documentos legales (ordenanzas, reglamentos)"""
        text = doc.text
        chunks = []
        
        # Dividir por artículos
        article_pattern = r'(Artículo\s+\d+[.\-\s]*[.:]*)'
        articles = re.split(article_pattern, text, flags=re.IGNORECASE)
        
        current_chunk = ""
        article_number = None
        
        for i, part in enumerate(articles):
            if re.match(article_pattern, part, re.IGNORECASE):
                # Guardar chunk anterior si existe
                if current_chunk.strip():
                    chunks.append(Document(
                        text=current_chunk.strip(),
                        metadata={
                            **doc.metadata, 
                            "section_type": "article",
                            "article_number": article_number
                        }
                    ))
                
                # Extraer número de artículo
                match = re.search(r'(\d+)', part)
                article_number = match.group(1) if match else None
                current_chunk = part
            else:
                current_chunk += part
        
        # Añadir último chunk
        if current_chunk.strip():
            chunks.append(Document(
                text=current_chunk.strip(),
                metadata={
                    **doc.metadata, 
                    "section_type": "article",
                    "article_number": article_number
                }
            ))
        
        return chunks if chunks else [doc]  # Fallback al documento original
    
    def _chunk_meeting_document(self, doc: Document) -> List[Document]:
        """Chunking especializado para actas de reuniones"""
        text = doc.text
        chunks = []
        
        # Dividir por puntos del día
        point_patterns = [
            r'(\d+[.\-\s]*[Pp]unto[:\s])',
            r'(Punto\s+\d+[:\s])',
            r'(\d+\.\s*[A-Z])'  # Patrón alternativo
        ]
        
        for pattern in point_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                points = re.split(pattern, text, flags=re.IGNORECASE)
                
                current_chunk = ""
                point_number = None
                
                for i, part in enumerate(points):
                    if re.match(pattern, part, re.IGNORECASE):
                        if current_chunk.strip():
                            chunks.append(Document(
                                text=current_chunk.strip(),
                                metadata={
                                    **doc.metadata, 
                                    "section_type": "agenda_point",
                                    "point_number": point_number
                                }
                            ))
                        
                        # Extraer número de punto
                        match = re.search(r'(\d+)', part)
                        point_number = match.group(1) if match else None
                        current_chunk = part
                    else:
                        current_chunk += part
                
                # Añadir último chunk
                if current_chunk.strip():
                    chunks.append(Document(
                        text=current_chunk.strip(),
                        metadata={
                            **doc.metadata, 
                            "section_type": "agenda_point",
                            "point_number": point_number
                        }
                    ))
                break
        
        return chunks if chunks else [doc]
    
    def _chunk_budget_document(self, doc: Document) -> List[Document]:
        """Chunking especializado para documentos presupuestarios"""
        text = doc.text
        chunks = []
        
        # Dividir por capítulos o partidas presupuestarias
        budget_pattern = r'(Capítulo\s+[IVXLC\d]+|Partida\s+\d+)'
        sections = re.split(budget_pattern, text, flags=re.IGNORECASE)
        
        current_chunk = ""
        section_id = None
        
        for i, part in enumerate(sections):
            if re.match(budget_pattern, part, re.IGNORECASE):
                if current_chunk.strip():
                    chunks.append(Document(
                        text=current_chunk.strip(),
                        metadata={
                            **doc.metadata, 
                            "section_type": "budget_section",
                            "section_id": section_id
                        }
                    ))
                
                section_id = part.strip()
                current_chunk = part
            else:
                current_chunk += part
        
        # Añadir último chunk
        if current_chunk.strip():
            chunks.append(Document(
                text=current_chunk.strip(),
                metadata={
                    **doc.metadata, 
                    "section_type": "budget_section",
                    "section_id": section_id
                }
            ))
        
        return chunks if chunks else [doc]