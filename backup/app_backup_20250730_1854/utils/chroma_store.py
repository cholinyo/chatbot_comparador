"""
ChromaDB Vector Store - Reemplazo optimizado de FAISS para administraciones locales
Diseñado específicamente para documentos municipales con metadatos enriquecidos
"""

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Configurar logging
logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """
    Vector store optimizado para administraciones locales con:
    - Persistencia automática
    - Metadatos enriquecidos para documentos municipales
    - Filtros avanzados por tipo documental
    - Trazabilidad completa
    """
    
    def __init__(self, collection_name: str = "municipal_docs"):
        self.collection_name = collection_name
        self.persist_directory = Path("vectorstore/chroma")
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Configurar ChromaDB con persistencia
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            logger.info(f"ChromaDB inicializado en: {self.persist_directory}")
        except Exception as e:
            logger.error(f"Error inicializando ChromaDB: {e}")
            raise
        
        # Modelo de embeddings (compatible con sistema existente)
        self.embedding_model = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Wrapper de LangChain para compatibilidad
        try:
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=str(self.persist_directory)
            )
            logger.info(f"Colección '{self.collection_name}' lista")
        except Exception as e:
            logger.error(f"Error creando vectorstore: {e}")
            raise
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None, ids: List[str] = None) -> List[str]:
        """
        Añadir documentos con metadatos enriquecidos para administración municipal
        
        Args:
            texts: Lista de textos a indexar
            metadatas: Metadatos enriquecidos por documento
            ids: IDs únicos opcionales
            
        Returns:
            Lista de IDs asignados
        """
        if not texts:
            logger.warning("No se proporcionaron textos para indexar")
            return []
        
        # Enriquecer metadatos con información de ingesta
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # Añadir metadatos de sistema
        enriched_metadatas = []
        for i, metadata in enumerate(metadatas):
            enriched = {
                **metadata,
                "ingestion_timestamp": datetime.now().isoformat(),
                "text_length": len(texts[i]),
                "collection": self.collection_name,
                "embedding_model": "all-MiniLM-L6-v2"
            }
            enriched_metadatas.append(enriched)
        
        try:
            # Usar LangChain para añadir documentos
            doc_ids = self.vectorstore.add_texts(
                texts=texts,
                metadatas=enriched_metadatas,
                ids=ids
            )
            logger.info(f"Añadidos {len(doc_ids)} documentos a la colección")
            return doc_ids
        except Exception as e:
            logger.error(f"Error añadiendo documentos: {e}")
            raise
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5, 
                         filter_metadata: Dict = None,
                         score_threshold: float = None) -> List[Dict]:
        """
        Búsqueda por similitud con filtros avanzados específicos para administración
        
        Args:
            query: Consulta de búsqueda
            k: Número de resultados
            filter_metadata: Filtros por metadatos (ej: {"document_type": "ordenanza"})
            score_threshold: Umbral mínimo de similitud
            
        Returns:
            Lista de documentos con metadata y scores
        """
        try:
            if filter_metadata:
                # Búsqueda con filtros
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                # Búsqueda estándar
                results = self.vectorstore.similarity_search(query, k=k)
            
            # Convertir a formato compatible con sistema existente
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "texto": doc.page_content,
                    "metadata": doc.metadata,
                    "fuente": doc.metadata.get("document_type", "general"),
                    "origen": doc.metadata.get("source_file", "desconocido"),
                    "fecha_ingesta": doc.metadata.get("ingestion_timestamp", ""),
                    "distancia": 0.0  # ChromaDB maneja scoring internamente
                })
            
            logger.info(f"Búsqueda completada: {len(formatted_results)} resultados para '{query[:50]}...'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda de similitud: {e}")
            return []
    
    def search_by_document_type(self, query: str, doc_type: str, k: int = 5) -> List[Dict]:
        """
        Búsqueda específica por tipo de documento municipal
        
        Args:
            query: Consulta
            doc_type: Tipo de documento (ordenanza, acta, resolucion, etc.)
            k: Número de resultados
        """
        filter_metadata = {"document_type": doc_type}
        return self.similarity_search(query, k, filter_metadata)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Estadísticas detalladas de la colección para dashboard administrativo
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            
            # Estadísticas básicas
            total_docs = collection.count()
            
            # Analizar metadatos (muestra de 1000 documentos)
            sample_size = min(1000, total_docs)
            if sample_size > 0:
                results = collection.get(limit=sample_size, include=["metadatas"])
                metadatas = results.get('metadatas', [])
                
                # Análisis de tipos de documentos
                doc_types = {}
                sources = {}
                unique_metadata_keys = set()
                
                for metadata in metadatas:
                    if metadata:
                        # Contar tipos de documentos
                        doc_type = metadata.get("document_type", "general")
                        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                        
                        # Contar fuentes
                        source = metadata.get("source_file", "desconocido")
                        sources[source] = sources.get(source, 0) + 1
                        
                        # Recopilar campos de metadatos únicos
                        unique_metadata_keys.update(metadata.keys())
            else:
                doc_types = {}
                sources = {}
                unique_metadata_keys = set()
            
            stats = {
                "total_documents": total_docs,
                "collection_name": self.collection_name,
                "document_types": doc_types,
                "top_sources": dict(sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]),
                "metadata_fields": list(unique_metadata_keys),
                "last_updated": datetime.now().isoformat(),
                "embedding_model": "all-Mini