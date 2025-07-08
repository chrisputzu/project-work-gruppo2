import os
import logging
import time
import random
import hashlib
import json
from typing import List, Dict, Any
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import SecretStr

from src.config import Config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processore per documenti PDF dei bandi"""
    
    def __init__(self):
        self.config = Config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self.embeddings = self._get_embeddings()
        self.vector_store = None
        
    def _get_embeddings(self):
        """Ottiene il modello di embedding appropriato"""
        if self.config.use_azure_embeddings():
            logger.info("Usando Azure OpenAI Embeddings dedicato")
            return AzureOpenAIEmbeddings(
                azure_endpoint=self.config.AZURE_EMBEDDING_ENDPOINT,
                api_key=SecretStr(self.config.AZURE_EMBEDDING_API_KEY) if self.config.AZURE_EMBEDDING_API_KEY else None,
                api_version=self.config.AZURE_API_VERSION,
                azure_deployment=self.config.AZURE_EMBEDDING_DEPLOYMENT_NAME,
                chunk_size=self.config.CHUNK_SIZE
            )
        elif self.config.use_azure_openai():
            logger.info("Usando Azure OpenAI Embeddings standard")
            return AzureOpenAIEmbeddings(
                azure_endpoint=self.config.AZURE_ENDPOINT,
                api_key=SecretStr(self.config.AZURE_API_KEY) if self.config.AZURE_API_KEY else None,
                api_version=self.config.AZURE_API_VERSION,
                azure_deployment=self.config.AZURE_EMBEDDING_DEPLOYMENT_NAME,
                chunk_size=self.config.CHUNK_SIZE
            )
        else:
            raise ValueError("Configurazione embeddings non valida")
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Carica un file PDF e lo converte in documenti"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Aggiungi metadati
            for doc in documents:
                doc.metadata.update({
                    'source': os.path.basename(file_path),
                    'file_path': file_path,
                    'file_type': 'pdf'
                })
            
            logger.info(f"Caricato PDF: {file_path} ({len(documents)} pagine)")
            return documents
            
        except Exception as e:
            logger.error(f"Errore nel caricamento del PDF {file_path}: {str(e)}")
            raise
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Processa i documenti dividendoli in chunks"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Creati {len(chunks)} chunks da {len(documents)} documenti")
            return chunks
            
        except Exception as e:
            logger.error(f"Errore nel processamento dei documenti: {str(e)}")
            raise
    
    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """Crea un vector store dai chunks"""
        try:
            if not self.embeddings:
                raise ValueError("Embeddings non inizializzati")
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            logger.info(f"Creato vector store con {len(chunks)} chunks")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Errore nella creazione del vector store: {str(e)}")
            raise
    
    def create_vector_store_batch(self, chunks: List[Document], progress_callback=None) -> FAISS:
        """Crea un vector store processando i chunks in batch per evitare rate limits"""
        try:
            if not self.embeddings:
                raise ValueError("Embeddings non inizializzati")
            
            if not chunks:
                raise ValueError("Nessun chunk da processare")
            
            # Se ci sono pochi chunks, usa il metodo normale
            if len(chunks) <= self.config.BATCH_SIZE:
                return self.create_vector_store(chunks)
            
            # logger.info(f"Processamento batch di {len(chunks)} chunks (batch size: {self.config.BATCH_SIZE})")
            
            # Dividi i chunks in batch
            batches = [chunks[i:i + self.config.BATCH_SIZE] 
                      for i in range(0, len(chunks), self.config.BATCH_SIZE)]
            
            logger.info(f"Creati {len(batches)} batch da processare")
            
            # Processa il primo batch per creare il vector store base
            first_batch = batches[0]
            if progress_callback:
                progress_callback(f"Processando batch 1/{len(batches)} ({len(first_batch)} chunks)...")
            
            vector_store = self._process_batch_with_retry(first_batch, batch_num=1, total_batches=len(batches))
            
            # Processa i batch rimanenti e combinali
            for i, batch in enumerate(batches[1:], 2):
                if progress_callback:
                    progress_callback(f"Processando batch {i}/{len(batches)} ({len(batch)} chunks)...")
                
                # Attesa tra batch per evitare rate limits
                time.sleep(self.config.BATCH_DELAY)
                
                batch_vector_store = self._process_batch_with_retry(batch, batch_num=i, total_batches=len(batches))
                
                # Combina i vector stores
                vector_store.merge_from(batch_vector_store)
                logger.info(f"Combinato batch {i}/{len(batches)}")
            
            self.vector_store = vector_store
            logger.info(f"Vector store finale creato con {len(chunks)} chunks totali")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Errore nella creazione del vector store batch: {str(e)}")
            raise
    
    def _process_batch_with_retry(self, batch_chunks: List[Document], batch_num: int, total_batches: int) -> FAISS:
        """Processa un singolo batch con retry automatico"""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                logger.info(f"Tentativo {attempt + 1}/{self.config.MAX_RETRIES} per batch {batch_num}/{total_batches}")
                return FAISS.from_documents(batch_chunks, self.embeddings)
                
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    if attempt < self.config.MAX_RETRIES - 1:
                        # Exponential backoff con jitter
                        delay = self.config.RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Rate limit hit per batch {batch_num}. Attesa di {delay:.1f} secondi...")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Rate limit persistente dopo {self.config.MAX_RETRIES} tentativi per batch {batch_num}")
                        raise
                else:
                    logger.error(f"Errore non rate-limit per batch {batch_num}: {str(e)}")
                    raise
        
        raise Exception(f"Impossibile processare batch {batch_num} dopo {self.config.MAX_RETRIES} tentativi")
    
    def _get_files_hash(self, file_paths: List[str]) -> str:
        """Calcola un hash basato sui file e le loro modifiche"""
        hash_data = []
        
        for file_path in file_paths:
            try:
                # Ottieni informazioni sul file
                file_stat = os.stat(file_path)
                file_info = {
                    'path': file_path,
                    'size': file_stat.st_size,
                    'mtime': file_stat.st_mtime,
                    'name': os.path.basename(file_path)
                }
                hash_data.append(file_info)
            except Exception as e:
                logger.warning(f"Impossibile ottenere stat per {file_path}: {e}")
                continue
        
        # Aggiungi configurazione del chunking
        config_data = {
            'chunk_size': self.config.CHUNK_SIZE,
            'chunk_overlap': self.config.CHUNK_OVERLAP,
            'batch_size': self.config.BATCH_SIZE
        }
        hash_data.append(config_data)
        
        # Calcola hash MD5
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def _get_vector_store_path(self, files_hash: str) -> str:
        """Ottiene il percorso del vector store per un hash specifico"""
        vector_store_dir = Path(self.config.VECTOR_STORE_DIR)
        vector_store_dir.mkdir(exist_ok=True)
        return str(vector_store_dir / f"vectorstore_{files_hash}")
    
    def _save_vector_store_metadata(self, files_hash: str, file_paths: List[str], chunks_count: int):
        """Salva i metadati del vector store"""
        metadata = {
            'files_hash': files_hash,
            'file_paths': file_paths,
            'chunks_count': chunks_count,
            'created_at': time.time(),
            'config': {
                'chunk_size': self.config.CHUNK_SIZE,
                'chunk_overlap': self.config.CHUNK_OVERLAP,
                'batch_size': self.config.BATCH_SIZE
            }
        }
        
        metadata_path = Path(self.config.VECTOR_STORE_DIR) / f"metadata_{files_hash}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadati salvati: {metadata_path}")
    
    def _load_vector_store_metadata(self, files_hash: str) -> Dict[str, Any]:
        """Carica i metadati del vector store"""
        metadata_path = Path(self.config.VECTOR_STORE_DIR) / f"metadata_{files_hash}.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _vector_store_exists(self, files_hash: str) -> bool:
        """Verifica se esiste un vector store per l'hash specificato"""
        vector_store_path = self._get_vector_store_path(files_hash)
        metadata_path = Path(self.config.VECTOR_STORE_DIR) / f"metadata_{files_hash}.json"
        
        return (Path(vector_store_path).exists() and 
                Path(f"{vector_store_path}.faiss").exists() and 
                Path(f"{vector_store_path}.pkl").exists() and
                metadata_path.exists())
    
    def _load_cached_vector_store(self, files_hash: str) -> FAISS:
        """Carica un vector store dalla cache"""
        vector_store_path = self._get_vector_store_path(files_hash)
        
        try:
            if not self.embeddings:
                raise ValueError("Embeddings non inizializzati")
            
            # Carica il vector store
            self.vector_store = FAISS.load_local(
                vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Carica i metadati
            metadata = self._load_vector_store_metadata(files_hash)
            
            logger.info(f"Vector store caricato dalla cache: {vector_store_path}")
            logger.info(f"Chunks: {metadata.get('chunks_count', 'N/A')}, Files: {len(metadata.get('file_paths', []))}")
            
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Errore nel caricamento del vector store cached: {str(e)}")
            raise
    
    def _save_vector_store_to_cache(self, files_hash: str, file_paths: List[str], chunks_count: int):
        """Salva il vector store nella cache"""
        if not self.vector_store:
            logger.warning("Nessun vector store da salvare")
            return
        
        vector_store_path = self._get_vector_store_path(files_hash)
        
        try:
            # Salva il vector store
            self.vector_store.save_local(vector_store_path)
            
            # Salva i metadati
            self._save_vector_store_metadata(files_hash, file_paths, chunks_count)
            
            logger.info(f"Vector store salvato nella cache: {vector_store_path}")
            
        except Exception as e:
            logger.error(f"Errore nel salvataggio del vector store: {str(e)}")
    
    def save_vector_store(self, path: str):
        """Salva il vector store su disco"""
        if self.vector_store:
            self.vector_store.save_local(path)
            logger.info(f"Vector store salvato in: {path}")
    
    def load_vector_store(self, path: str) -> FAISS:
        """Carica un vector store da disco"""
        try:
            if not self.embeddings:
                raise ValueError("Embeddings non inizializzati")
            self.vector_store = FAISS.load_local(path, self.embeddings)
            logger.info(f"Vector store caricato da: {path}")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Errore nel caricamento del vector store: {str(e)}")
            raise
    
    def process_multiple_files(self, file_paths: List[str]) -> FAISS:
        """Processa multipli file PDF"""
        all_chunks = []
        
        for file_path in file_paths:
            try:
                documents = self.load_pdf(file_path)
                chunks = self.process_documents(documents)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Errore nel processamento di {file_path}: {str(e)}")
                continue
        
        if all_chunks:
            return self.create_vector_store(all_chunks)
        else:
            raise ValueError("Nessun documento processato con successo")
    
    def process_multiple_files_batch(self, file_paths: List[str], progress_callback=None) -> FAISS:
        """Processa multipli file PDF usando processamento batch per evitare rate limits"""
        # Calcola hash dei file
        files_hash = self._get_files_hash(file_paths)
        
        # Controlla se esiste già un vector store in cache
        if self._vector_store_exists(files_hash):
            if progress_callback:
                progress_callback("🔍 Embeddings trovati in cache, caricamento in corso...")
            
            logger.info(f"Caricamento vector store dalla cache (hash: {files_hash})")
            
            try:
                vector_store = self._load_cached_vector_store(files_hash)
                
                if progress_callback:
                    metadata = self._load_vector_store_metadata(files_hash)
                    chunks_count = metadata.get('chunks_count', 'N/A')
                    progress_callback(f"✅ Embeddings caricati dalla cache! {chunks_count} chunks pronti")
                
                return vector_store
                
            except Exception as e:
                logger.warning(f"Errore nel caricamento dalla cache: {e}. Procedo con il calcolo...")
                if progress_callback:
                    progress_callback("⚠️ Errore nel caricamento cache, ricalcolo embeddings...")
        
        # Se non esiste in cache, processa normalmente
        all_chunks = []
        total_files = len(file_paths)
        
        # Fase 1: Caricamento e chunking dei file
        if progress_callback:
            progress_callback("📄 Caricamento e divisione in chunks dei documenti...")
        
        for i, file_path in enumerate(file_paths):
            try:
                if progress_callback:
                    progress_callback(f"Caricando file {i+1}/{total_files}: {os.path.basename(file_path)}")
                
                documents = self.load_pdf(file_path)
                chunks = self.process_documents(documents)
                all_chunks.extend(chunks)
                
                logger.info(f"File {i+1}/{total_files} processato: {len(chunks)} chunks da {os.path.basename(file_path)}")
                
            except Exception as e:
                logger.error(f"Errore nel processamento di {file_path}: {str(e)}")
                if progress_callback:
                    progress_callback(f"⚠️ Errore nel file {os.path.basename(file_path)}: {str(e)}")
                continue
        
        if not all_chunks:
            raise ValueError("Nessun documento processato con successo")
        
        logger.info(f"Totale chunks generati: {len(all_chunks)} da {total_files} file")
        
        # Fase 2: Creazione vector store con batch processing
        if progress_callback:
            progress_callback(f"💾 Creazione embeddings per {len(all_chunks)} chunks...")
        
        vector_store = self.create_vector_store_batch(all_chunks, progress_callback)
        
        # Fase 3: Salvataggio nella cache
        if progress_callback:
            progress_callback("💾 Salvataggio embeddings in cache...")
        
        try:
            self._save_vector_store_to_cache(files_hash, file_paths, len(all_chunks))
            
            if progress_callback:
                progress_callback("✅ Embeddings salvati in cache per utilizzi futuri!")
            
        except Exception as e:
            logger.error(f"Errore nel salvataggio cache: {e}")
            if progress_callback:
                progress_callback(f"⚠️ Errore nel salvataggio cache: {e}")
        
        return vector_store
    
    def extract_document_info(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Estrae informazioni strutturate dai documenti"""
        doc_info = []
        
        for doc in documents:
            info = {
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 0),
                'content_length': len(doc.page_content),
                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            doc_info.append(info)
        
        return doc_info