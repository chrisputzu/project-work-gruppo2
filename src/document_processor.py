import os
import logging
import time
import random
import hashlib
import json
from typing import List, Dict, Any
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import SecretStr

# Importazioni per conversione PDF->Markdown
import fitz  # PyMuPDF
import pymupdf4llm

from src.config import Config

logger = logging.getLogger(__name__)

class EnhancedDocumentProcessor:
    """Processore migliorato per documenti PDF con conversione in Markdown"""
    
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
    
    def pdf_to_markdown(self, pdf_path: str) -> str:
        """Converte un PDF in formato Markdown usando pymupdf4llm"""
        try:
            logger.info(f"Convertendo PDF in Markdown: {pdf_path}")
            
            # Usa pymupdf4llm per conversione ottimizzata per LLM
            markdown_text = pymupdf4llm.to_markdown(pdf_path)
            
            logger.info(f"Conversione completata. Lunghezza Markdown: {len(markdown_text)} caratteri")
            return markdown_text
            
        except Exception as e:
            logger.error(f"Errore nella conversione PDF->Markdown per {pdf_path}: {str(e)}")
            # Fallback a estrazione testo semplice
            return self._fallback_pdf_extraction(pdf_path)
    
    def _fallback_pdf_extraction(self, pdf_path: str) -> str:
        """Estrazione di fallback usando PyMuPDF base"""
        try:
            logger.warning(f"Usando estrazione di fallback per {pdf_path}")
            
            doc = fitz.open(pdf_path)
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Aggiungi separatore di pagina in markdown
                text_content.append(f"\n---\n**Pagina {page_num + 1}**\n\n{text}")
            
            doc.close()
            return "\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Errore anche nell'estrazione di fallback per {pdf_path}: {str(e)}")
            raise
    
    def save_markdown(self, markdown_content: str, pdf_path: str) -> str:
        """Salva il contenuto Markdown su file"""
        try:
            # Crea directory per i markdown
            markdown_dir = Path("markdown_cache")
            markdown_dir.mkdir(exist_ok=True)
            
            # Nome file markdown basato sul PDF
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            markdown_path = markdown_dir / f"{pdf_name}.md"
            
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Markdown salvato: {markdown_path}")
            return str(markdown_path)
            
        except Exception as e:
            logger.error(f"Errore nel salvataggio markdown: {str(e)}")
            raise
    
    def load_pdf_as_markdown(self, file_path: str) -> List[Document]:
        """Carica un PDF convertendolo prima in Markdown"""
        try:
            # Converti PDF in Markdown
            markdown_content = self.pdf_to_markdown(file_path)
            
            # Salva il markdown (opzionale, per debug/cache)
            markdown_path = self.save_markdown(markdown_content, file_path)
            
            # Crea documento LangChain dal markdown
            document = Document(
                page_content=markdown_content,
                metadata={
                    'source': os.path.basename(file_path),
                    'file_path': file_path,
                    'file_type': 'pdf',
                    'markdown_path': markdown_path,
                    'conversion_method': 'pymupdf4llm',
                    'total_length': len(markdown_content)
                }
            )
            
            logger.info(f"PDF caricato come Markdown: {file_path}")
            return [document]
            
        except Exception as e:
            logger.error(f"Errore nel caricamento del PDF {file_path}: {str(e)}")
            raise
    
    def load_pdf_as_markdown_pages(self, file_path: str) -> List[Document]:
        """Carica un PDF convertendolo in Markdown mantenendo la separazione per pagine"""
        try:
            logger.info(f"Caricando PDF per pagine come Markdown: {file_path}")
            
            # Apri il PDF
            doc = fitz.open(file_path)
            documents = []
            
            for page_num in range(len(doc)):
                # Estrai una pagina alla volta
                page = doc.load_page(page_num)
                
                # Converti la singola pagina in markdown usando pymupdf4llm
                try:
                    # Crea un PDF temporaneo con solo questa pagina
                    temp_doc = fitz.open()
                    temp_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    
                    temp_path = f"temp_page_{page_num}.pdf"
                    temp_doc.save(temp_path)
                    temp_doc.close()
                    
                    # Converti la pagina in markdown
                    page_markdown = pymupdf4llm.to_markdown(temp_path)
                    
                    # Rimuovi il file temporaneo
                    os.remove(temp_path)
                    
                except Exception as e:
                    logger.warning(f"Errore conversione pagina {page_num}, uso fallback: {e}")
                    # Fallback a testo semplice
                    page_markdown = page.get_text()
                
                # Crea documento per la pagina
                page_document = Document(
                    page_content=page_markdown,
                    metadata={
                        'source': os.path.basename(file_path),
                        'file_path': file_path,
                        'file_type': 'pdf',
                        'page': page_num + 1,
                        'total_pages': len(doc),
                        'conversion_method': 'pymupdf4llm_per_page',
                        'page_length': len(page_markdown)
                    }
                )
                
                documents.append(page_document)
            
            doc.close()
            
            logger.info(f"PDF caricato: {file_path} ({len(documents)} pagine in Markdown)")
            return documents
            
        except Exception as e:
            logger.error(f"Errore nel caricamento del PDF per pagine {file_path}: {str(e)}")
            raise
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Processa i documenti Markdown dividendoli in chunks"""
        try:
            # Configura il text splitter per Markdown
            markdown_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                separators=[
                    "\n## ",  # Headers H2
                    "\n### ", # Headers H3
                    "\n#### ", # Headers H4
                    "\n\n",   # Paragrafi
                    "\n",     # Righe
                    ". ",     # Frasi
                    " ",      # Parole
                    ""        # Caratteri
                ]
            )
            
            chunks = markdown_splitter.split_documents(documents)
            
            # Arricchisci i metadati dei chunks
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'chunk_length': len(chunk.page_content),
                    'content_type': 'markdown'
                })
            
            logger.info(f"Creati {len(chunks)} chunks Markdown da {len(documents)} documenti")
            return chunks
            
        except Exception as e:
            logger.error(f"Errore nel processamento dei documenti Markdown: {str(e)}")
            raise
    
    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """Crea un vector store dai chunks Markdown"""
        try:
            if not self.embeddings:
                raise ValueError("Embeddings non inizializzati")
            
            logger.info(f"Creando vector store da {len(chunks)} chunks Markdown")
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            logger.info(f"Vector store creato con {len(chunks)} chunks Markdown")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Errore nella creazione del vector store: {str(e)}")
            raise
    
    def create_vector_store_batch(self, chunks: List[Document], progress_callback=None) -> FAISS:
        """Crea un vector store processando i chunks Markdown in batch"""
        try:
            if not self.embeddings:
                raise ValueError("Embeddings non inizializzati")
            
            if not chunks:
                raise ValueError("Nessun chunk da processare")
            
            # Se ci sono pochi chunks, usa il metodo normale
            if len(chunks) <= self.config.BATCH_SIZE:
                return self.create_vector_store(chunks)
            
            logger.info(f"Processamento batch di {len(chunks)} chunks Markdown (batch size: {self.config.BATCH_SIZE})")
            
            # Dividi i chunks in batch
            batches = [chunks[i:i + self.config.BATCH_SIZE] 
                      for i in range(0, len(chunks), self.config.BATCH_SIZE)]
            
            logger.info(f"Creati {len(batches)} batch da processare")
            
            # Processa il primo batch per creare il vector store base
            first_batch = batches[0]
            if progress_callback:
                progress_callback('Creando embeddings dal contenuto Markdown...')
            
            vector_store = self._process_batch_with_retry(first_batch, batch_num=1, total_batches=len(batches))
            
            # Processa i batch rimanenti e combinali
            for i, batch in enumerate(batches[1:], 2):
                if progress_callback:
                    progress_callback(f"Caricamento...")
                
                # Attesa tra batch per evitare rate limits
                time.sleep(self.config.BATCH_DELAY)
                
                batch_vector_store = self._process_batch_with_retry(batch, batch_num=i, total_batches=len(batches))
                
                # Combina i vector stores
                vector_store.merge_from(batch_vector_store)
                logger.info(f"Combinato batch {i}/{len(batches)}")
            
            self.vector_store = vector_store
            logger.info(f"Vector store finale creato con {len(chunks)} chunks Markdown totali")
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
    
    def process_multiple_files_markdown(self, file_paths: List[str], split_by_pages: bool = False, progress_callback=None) -> FAISS:
        """Processa multipli file PDF convertendoli in Markdown"""
        # Calcola hash dei file
        files_hash = self._get_files_hash(file_paths)
        files_hash += "_markdown"  # Distingui dalla cache normale
        
        # Controlla se esiste già un vector store in cache
        if self._vector_store_exists(files_hash):
            if progress_callback:
                progress_callback("🔍 Embeddings Markdown trovati in cache, caricamento...")
            
            logger.info(f"Caricamento vector store Markdown dalla cache (hash: {files_hash})")
            
            try:
                vector_store = self._load_cached_vector_store(files_hash)
                
                if progress_callback:
                    metadata = self._load_vector_store_metadata(files_hash)
                    chunks_count = metadata.get('chunks_count', 'N/A')
                    progress_callback(f"✅ Embeddings Markdown caricati dalla cache! {chunks_count} chunks pronti")
                
                return vector_store
                
            except Exception as e:
                logger.warning(f"Errore nel caricamento dalla cache: {e}. Procedo con conversione...")
                if progress_callback:
                    progress_callback("⚠️ Errore cache, riconversione in Markdown...")
        
        # Se non esiste in cache, processa normalmente
        all_chunks = []
        total_files = len(file_paths)
        
        # Fase 1: Conversione PDF -> Markdown e chunking
        if progress_callback:
            progress_callback("📄 Conversione PDF -> Markdown in corso...")
        
        for i, file_path in enumerate(file_paths):
            try:
                if progress_callback:
                    progress_callback(f"Convertendo {i+1}/{total_files}: {os.path.basename(file_path)}")
                
                # Carica PDF come Markdown
                if split_by_pages:
                    documents = self.load_pdf_as_markdown_pages(file_path)
                else:
                    documents = self.load_pdf_as_markdown(file_path)
                
                # Processa i documenti in chunks
                chunks = self.process_documents(documents)
                all_chunks.extend(chunks)
                
                logger.info(f"File {i+1}/{total_files} convertito: {len(chunks)} chunks da {os.path.basename(file_path)}")
                
            except Exception as e:
                logger.error(f"Errore nella conversione di {file_path}: {str(e)}")
                if progress_callback:
                    progress_callback(f"⚠️ Errore nel file {os.path.basename(file_path)}: {str(e)}")
                continue
        
        if not all_chunks:
            raise ValueError("Nessun documento convertito con successo")
        
        logger.info(f"Totale chunks Markdown generati: {len(all_chunks)} da {total_files} file")
        
        # Fase 2: Creazione vector store con batch processing
        if progress_callback:
            progress_callback(f"💾 Creazione embeddings da {len(all_chunks)} chunks Markdown...")
        
        vector_store = self.create_vector_store_batch(all_chunks, progress_callback)
        
        # Fase 3: Salvataggio nella cache
        if progress_callback:
            progress_callback("💾 Salvataggio embeddings Markdown in cache...")
        
        try:
            self._save_vector_store_to_cache(files_hash, file_paths, len(all_chunks))
            
            if progress_callback:
                progress_callback("✅ Embeddings Markdown salvati in cache!")
            
        except Exception as e:
            logger.error(f"Errore nel salvataggio cache: {e}")
            if progress_callback:
                progress_callback(f"⚠️ Errore nel salvataggio cache: {e}")
        
        return vector_store
    
    # Metodi di cache riutilizzati dalla classe originale
    def _get_files_hash(self, file_paths: List[str]) -> str:
        """Calcola un hash basato sui file e le loro modifiche"""
        hash_data = []
        
        for file_path in file_paths:
            try:
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
        
        config_data = {
            'chunk_size': self.config.CHUNK_SIZE,
            'chunk_overlap': self.config.CHUNK_OVERLAP,
            'batch_size': self.config.BATCH_SIZE
        }
        hash_data.append(config_data)
        
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
            'conversion_method': 'markdown',
            'config': {
                'chunk_size': self.config.CHUNK_SIZE,
                'chunk_overlap': self.config.CHUNK_OVERLAP,
                'batch_size': self.config.BATCH_SIZE
            }
        }
        
        metadata_path = Path(self.config.VECTOR_STORE_DIR) / f"metadata_{files_hash}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadati Markdown salvati: {metadata_path}")
    
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
            
            self.vector_store = FAISS.load_local(
                vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            metadata = self._load_vector_store_metadata(files_hash)
            
            logger.info(f"Vector store Markdown caricato dalla cache: {vector_store_path}")
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
            self.vector_store.save_local(vector_store_path)
            self._save_vector_store_metadata(files_hash, file_paths, chunks_count)
            logger.info(f"Vector store Markdown salvato nella cache: {vector_store_path}")
            
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
            self.vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Vector store caricato da: {path}")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Errore nel caricamento del vector store: {str(e)}")
            raise
    
    def extract_document_info(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Estrae informazioni strutturate dai documenti Markdown"""
        doc_info = []
        
        for doc in documents:
            info = {
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 0),
                'content_length': len(doc.page_content),
                'content_type': doc.metadata.get('content_type', 'markdown'),
                'conversion_method': doc.metadata.get('conversion_method', 'unknown'),
                'markdown_path': doc.metadata.get('markdown_path', ''),
                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            doc_info.append(info)
        
        return doc_info