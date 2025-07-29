# app/services/ingest_documents_improved.py
import os
import logging
import json
import pickle
import hashlib
import numpy as np
import faiss
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from app.utils import doc_loader

# Configuración
CONFIG_PATH = os.path.join("app", "config", "settings.json")
VECTOR_DIR = os.path.join("vectorstore", "documents")
LOG_PATH = os.path.join("logs", "ingestion_detailed.log")
CHECKSUM_PATH = os.path.join(VECTOR_DIR, "document_checksums.json")

# Crear carpetas necesarias
os.makedirs("logs", exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# Configurar logging detallado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentIngestor:
    def __init__(self):
        self.modelo = SentenceTransformer("all-MiniLM-L6-v2")
        self.checksums = self.load_checksums()
        self.stats = {
            "procesados": 0,
            "nuevos": 0,
            "actualizados": 0,
            "errores": 0,
            "fragmentos_totales": 0
        }
        
    def load_checksums(self):
        """Cargar checksums de documentos procesados previamente"""
        try:
            if os.path.exists(CHECKSUM_PATH):
                with open(CHECKSUM_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"No se pudieron cargar checksums: {e}")
        return {}
    
    def save_checksums(self):
        """Guardar checksums de documentos procesados"""
        try:
            with open(CHECKSUM_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.checksums, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando checksums: {e}")
    
    def calculate_file_checksum(self, file_path):
        """Calcular checksum de un archivo"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"Error calculando checksum para {file_path}: {e}")
            return None
    
    def needs_processing(self, file_path):
        """Verificar si un documento necesita ser procesado"""
        file_str = str(file_path)
        current_checksum = self.calculate_file_checksum(file_path)
        
        if current_checksum is None:
            return False
            
        stored_checksum = self.checksums.get(file_str, {}).get('checksum')
        
        if stored_checksum != current_checksum:
            logger.info(f"Documento modificado detectado: {file_path}")
            return True
        
        logger.debug(f"Documento sin cambios: {file_path}")
        return False
    
    def validate_document(self, doc_content):
        """Validar y limpiar contenido del documento"""
        if not doc_content or len(doc_content.strip()) < 50:
            return None
            
        # Normalizar encoding
        try:
            # Limpiar caracteres problemáticos
            cleaned = doc_content.encode('utf-8', errors='ignore').decode('utf-8')
            # Normalizar espacios en blanco
            cleaned = ' '.join(cleaned.split())
            return cleaned
        except Exception as e:
            logger.error(f"Error validando documento: {e}")
            return None
    
    def create_semantic_chunks(self, text, max_chars=500, overlap=50):
        """Crear fragmentos semánticos con overlap"""
        if not text:
            return []
            
        # Dividir por párrafos primero
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Si el párrafo es muy largo, dividir por oraciones
            if len(paragraph) > max_chars:
                sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_chars:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
            else:
                # Añadir párrafo completo si cabe
                if len(current_chunk) + len(paragraph) <= max_chars:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
        
        # Añadir último chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Crear overlap entre chunks
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            overlapped_chunks.append(chunk)
            
            # Añadir overlap con el siguiente chunk
            if i < len(chunks) - 1 and overlap > 0:
                next_chunk_start = chunks[i + 1][:overlap]
                overlap_chunk = chunk[-overlap:] + " ... " + next_chunk_start
                overlapped_chunks.append(overlap_chunk)
        
        return overlapped_chunks
    
    def process_document_with_retry(self, doc_path, max_retries=3):
        """Procesar documento con reintentos"""
        for attempt in range(max_retries):
            try:
                return self.process_single_document(doc_path)
            except Exception as e:
                logger.warning(f"Intento {attempt + 1} fallido para {doc_path}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Falló procesamiento de {doc_path} después de {max_retries} intentos")
                    self.stats["errores"] += 1
                    return None
        return None
    
    def process_single_document(self, doc_path):
        """Procesar un documento individual"""
        logger.info(f"Procesando: {doc_path}")
        
        # Verificar si necesita procesamiento
        if not self.needs_processing(doc_path):
            return None
        
        # Leer contenido según extensión
        doc_path_obj = Path(doc_path)
        doc_path_str = str(doc_path)  # Convertir a string
        try:
            if doc_path_obj.suffix.lower() == ".txt":
                content = doc_loader.leer_txt(doc_path_str)
            elif doc_path_obj.suffix.lower() == ".docx":
                content = doc_loader.leer_docx(doc_path_str)
            elif doc_path_obj.suffix.lower() == ".pdf":
                content = doc_loader.leer_pdf(doc_path_str)
            elif doc_path_obj.suffix.lower() == ".html":
                content = doc_loader.leer_html(doc_path_str)
            else:
                logger.warning(f"Formato no soportado: {doc_path}")
                return None
        except Exception as e:
            logger.error(f"Error leyendo {doc_path}: {e}")
            raise
        
        # Validar contenido
        content = self.validate_document(content)
        if not content:
            logger.warning(f"Documento vacío o inválido: {doc_path}")
            return None
        
        # Crear fragmentos semánticos
        chunks = self.create_semantic_chunks(content)
        if not chunks:
            logger.warning(f"No se pudieron crear fragmentos para: {doc_path}")
            return None
        
        # Generar embeddings
        try:
            embeddings = self.modelo.encode(chunks, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error generando embeddings para {doc_path}: {e}")
            raise
        
        # Crear metadatos enriquecidos
        metadatos = []
        for i, chunk in enumerate(chunks):
            metadatos.append({
                "documento": doc_path_obj.name,
                "ruta_completa": str(doc_path),
                "fragmento_id": i,
                "texto": chunk,
                "longitud": len(chunk),
                "fecha_procesamiento": datetime.now().isoformat(),
                "origen": "documento",
                "checksum": self.calculate_file_checksum(doc_path)
            })
        
        # Actualizar checksum
        self.checksums[str(doc_path)] = {
            "checksum": self.calculate_file_checksum(doc_path),
            "fecha_procesamiento": datetime.now().isoformat(),
            "num_fragmentos": len(chunks)
        }
        
        # Actualizar estadísticas
        self.stats["procesados"] += 1
        self.stats["fragmentos_totales"] += len(chunks)
        
        logger.info(f"✅ Procesado {doc_path}: {len(chunks)} fragmentos")
        
        return {
            "embeddings": embeddings,
            "metadatos": metadatos,
            "fragmentos": chunks
        }
    
    def cargar_config(self):
        """Cargar configuración"""
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def save_vectorstore(self, all_embeddings, all_metadatos, all_fragmentos):
        """Guardar vectorstore con validación"""
        try:
            if not all_embeddings:
                logger.warning("No hay embeddings para guardar")
                return False
            
            # Crear índice FAISS
            embeddings_np = np.array(all_embeddings).astype("float32")
            index = faiss.IndexFlatL2(embeddings_np.shape[1])
            index.add(embeddings_np)
            
            # Guardar archivos
            np.save(os.path.join(VECTOR_DIR, "embeddings.npy"), embeddings_np)
            faiss.write_index(index, os.path.join(VECTOR_DIR, "index.faiss"))
            
            with open(os.path.join(VECTOR_DIR, "fragmentos.pkl"), "wb") as f:
                pickle.dump(all_fragmentos, f)
            
            with open(os.path.join(VECTOR_DIR, "metadatos.pkl"), "wb") as f:
                pickle.dump(all_metadatos, f)
            
            logger.info(f"✅ Vectorstore guardado: {len(all_embeddings)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando vectorstore: {e}")
            return False
    
    def run_ingestion(self):
        """Ejecutar proceso completo de ingesta"""
        logger.info("🚀 Iniciando ingesta mejorada de documentos")
        start_time = datetime.now()
        
        try:
            # Cargar configuración
            config = self.cargar_config()
            carpetas = config.get("document_folders", [])
            
            if not carpetas:
                logger.warning("No hay carpetas configuradas")
                return False
            
            # Recopilar todos los documentos
            documentos_paths = []
            for carpeta in carpetas:
                carpeta_path = Path(carpeta)
                if carpeta_path.exists():
                    for ext in [".txt", ".pdf", ".docx", ".html"]:
                        documentos_paths.extend(carpeta_path.rglob(f"*{ext}"))
            
            logger.info(f"📁 Encontrados {len(documentos_paths)} documentos")
            
            # Procesar documentos
            all_embeddings = []
            all_metadatos = []
            all_fragmentos = []
            
            for doc_path in tqdm(documentos_paths, desc="Procesando documentos"):
                if doc_path.name.startswith("~$"):  # Archivos temporales
                    continue
                    
                result = self.process_document_with_retry(doc_path)
                if result:
                    all_embeddings.extend(result["embeddings"])
                    all_metadatos.extend(result["metadatos"])
                    all_fragmentos.extend(result["fragmentos"])
            
            # Guardar resultados
            if all_embeddings:
                success = self.save_vectorstore(all_embeddings, all_metadatos, all_fragmentos)
                if success:
                    self.save_checksums()
            
            # Estadísticas finales
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("🎉 Ingesta completada")
            logger.info(f"📊 Estadísticas:")
            logger.info(f"   - Documentos procesados: {self.stats['procesados']}")
            logger.info(f"   - Fragmentos totales: {self.stats['fragmentos_totales']}")
            logger.info(f"   - Errores: {self.stats['errores']}")
            logger.info(f"   - Tiempo total: {duration:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en ingesta: {e}")
            return False

def main():
    """Función principal"""
    ingestor = DocumentIngestor()
    return ingestor.run_ingestion()

if __name__ == "__main__":
    main()