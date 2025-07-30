"""
ChromaDB Vector Store - Reemplazo optimizado de FAISS para administraciones locales
Integración con LangChain para máxima compatibilidad
"""
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import os
import logging
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper para integrar SentenceTransformers con LangChain"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embebida múltiples documentos"""
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embebida una consulta"""
        return self.model.encode([text])[0].tolist()

class ChromaVectorStore:
    """Vector store optimizado para administraciones locales con ChromaDB"""
    
    def __init__(self, collection_name: str = "admin_local_docs"):
        self.collection_name = collection_name
        self.persist_directory = "vectorstore/chroma"
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Configurar ChromaDB con persistencia
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Embeddings
        self.embeddings = SentenceTransformerEmbeddings()
        
        # LangChain Chroma wrapper
        try:
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info(f"✅ ChromaDB inicializado: {collection_name}")
        except Exception as e:
            logger.error(f"❌ Error inicializando ChromaDB: {e}")
            raise
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None) -> List[str]:
        """
        Añadir documentos con metadatos enriquecidos
        
        Args:
            texts: Lista de textos a indexar
            metadatas: Lista de diccionarios con metadatos
            
        Returns:
            Lista de IDs de documentos añadidos
        """
        if not texts:
            logger.warning("⚠️ No hay textos para añadir")
            return []
            
        # Generar metadatos por defecto si no se proporcionan
        if metadatas is None:
            metadatas = [{"added_at": datetime.now().isoformat()} for _ in texts]
        else:
            # Enriquecer metadatos existentes
            for metadata in metadatas:
                if "added_at" not in metadata:
                    metadata["added_at"] = datetime.now().isoformat()
                if "id" not in metadata:
                    metadata["id"] = str(uuid.uuid4())
        
        try:
            ids = self.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            logger.info(f"✅ Añadidos {len(ids)} documentos a ChromaDB")
            return ids
        except Exception as e:
            logger.error(f"❌ Error añadiendo documentos: {e}")
            return []
    
    def similarity_search(self, query: str, k: int = 5, filter_metadata: Dict = None) -> List[Dict]:
        """
        Búsqueda por similitud con filtros avanzados
        
        Args:
            query: Consulta de búsqueda
            k: Número de resultados a devolver
            filter_metadata: Filtros para metadatos
            
        Returns:
            Lista de documentos encontrados con metadatos
        """
        try:
            if filter_metadata:
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                results = self.vectorstore.similarity_search(query, k=k)
            
            # Convertir a formato estándar del sistema
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "texto": doc.page_content,
                    "metadata": doc.metadata,
                    "fuente": doc.metadata.get("document_type", "general"),
                    "origen": doc.metadata.get("origen", "unknown"),
                    "distancia": 0.0  # ChromaDB no expone distancia directamente
                })
            
            logger.info(f"🔍 Búsqueda completada: {len(formatted_results)} resultados")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Error en búsqueda: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la colección"""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            # Obtener muestra de metadatos para análisis
            sample_results = collection.get(limit=min(100, count))
            metadata_fields = set()
            
            for metadata in sample_results.get('metadatas', []):
                if metadata:
                    metadata_fields.update(metadata.keys())
            
            stats = {
                "total_documents": count,
                "collection_name": self.collection_name,
                "metadata_fields": list(metadata_fields),
                "persist_directory": self.persist_directory
            }
            
            logger.info(f"📊 Estadísticas: {count} documentos, {len(metadata_fields)} campos metadata")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas: {e}")
            return {
                "total_documents": 0,
                "collection_name": self.collection_name,
                "metadata_fields": [],
                "error": str(e)
            }
    
    def delete_collection(self):
        """Eliminar toda la colección (usar con cuidado)"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"🗑️ Colección {self.collection_name} eliminada")
        except Exception as e:
            logger.error(f"❌ Error eliminando colección: {e}")
    
    def search_by_metadata(self, metadata_filter: Dict, limit: int = 50) -> List[Dict]:
        """Buscar documentos solo por metadatos (sin query semántica)"""
        try:
            collection = self.client.get_collection(self.collection_name)
            results = collection.get(
                where=metadata_filter,
                limit=limit
            )
            
            formatted_results = []
            for i, doc_id in enumerate(results.get('ids', [])):
                formatted_results.append({
                    "id": doc_id,
                    "texto": results['documents'][i] if results.get('documents') else "",
                    "metadata": results['metadatas'][i] if results.get('metadatas') else {}
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Error en búsqueda por metadatos: {e}")
            return []

# Instancia global (patrón singleton)
_chroma_store_instance = None

def get_chroma_store(collection_name: str = "admin_local_docs") -> ChromaVectorStore:
    """Obtener instancia única de ChromaDB"""
    global _chroma_store_instance
    if _chroma_store_instance is None:
        _chroma_store_instance = ChromaVectorStore(collection_name)
    return _chroma_store_instance