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
                progress_callback('Creando gli embeddings...')
            
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
            progress_callback("Conversione PDF -> Markdown in corso...")
        
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
            progress_callback(f"Creazione embeddings da {len(all_chunks)} chunks Markdown...")
        
        vector_store = self.create_vector_store_batch(all_chunks, progress_callback)
        
        # Fase 3: Salvataggio nella cache
        if progress_callback:
            progress_callback("Salvataggio embeddings Markdown in cache...")
        
        try:
            self._save_vector_store_to_cache(files_hash, file_paths, len(all_chunks))
            
            if progress_callback:
                progress_callback("Embeddings Markdown salvati in cache!")
            
        except Exception as e:
            logger.error(f"Errore nel salvataggio cache: {e}")
            if progress_callback:
                progress_callback(f"Errore nel salvataggio cache: {e}")
        
        return vector_store
    
    def process_and_add_files(self, file_paths: List[str], split_by_pages: bool = False, progress_callback=None) -> FAISS:
        """
        Processa i file e li aggiunge a un vector store globale incrementale.
        Verifica quali file sono già stati vettorizzati per evitare duplicazioni.
        
        Args:
            file_paths: Lista dei percorsi dei file da processare
            split_by_pages: Se True, separa il documento per pagine
            progress_callback: Callback per aggiornare l'interfaccia utente
            
        Returns:
            Il vector store aggiornato con i nuovi file
        """
        if not file_paths:
            raise ValueError("Nessun file da processare")
            
        # Directory per il vector store globale
        vector_store_dir = Path(self.config.VECTOR_STORE_DIR)
        vector_store_dir.mkdir(exist_ok=True)
        global_vs_path = str(vector_store_dir / "global_vectorstore")
        global_meta_path = vector_store_dir / "global_metadata.json"
        
        # Controlla se esiste già un vector store globale
        global_vs_exists = (Path(global_vs_path).exists() and 
                           Path(f"{global_vs_path}.faiss").exists() and 
                           Path(f"{global_vs_path}.pkl").exists() and
                           global_meta_path.exists())
        
        # Carica i metadati globali o inizializza un nuovo dizionario
        if global_vs_exists:
            try:
                with open(global_meta_path, 'r', encoding='utf-8') as f:
                    global_metadata = json.load(f)
                processed_files = global_metadata.get('processed_files', {})
                filename_index = global_metadata.get('filename_index', {})
                
                if progress_callback:
                    progress_callback(f"🔍 Vector store globale trovato con {len(processed_files)} file")
                logger.info(f"Vector store globale trovato con {len(processed_files)} file")
            except Exception as e:
                logger.error(f"Errore nel caricamento dei metadati globali: {str(e)}")
                processed_files = {}
                filename_index = {}
                global_vs_exists = False
        else:
            processed_files = {}
            filename_index = {}
            if progress_callback:
                progress_callback("🆕 Creazione nuovo vector store globale")
            logger.info("Creazione nuovo vector store globale")
        
        # Filtra i file già processati
        new_files = []
        skipped_files = []
        
        for file_path in file_paths:
            try:
                # Genera hash basato su contenuto e data modifica per questo file
                file_hash = self._get_file_hash(file_path)
                file_name = os.path.basename(file_path)
                
                # Controllo basato sull'hash
                hash_exists = file_hash in processed_files and processed_files[file_hash]['path'] == file_path
                
                # Controllo aggiuntivo basato sul nome del file
                name_exists = False
                if filename_index and file_name in filename_index:
                    name_exists = True
                else:
                    # Cerchiamo nel modo tradizionale se l'indice non esiste o il file non è indicizzato
                    for existing_hash, file_info in processed_files.items():
                        if file_info.get('filename') == file_name:
                            name_exists = True
                            break
                
                if hash_exists or name_exists:
                    skipped_files.append(file_name)
                    logger.info(f"File {file_name} già elaborato, saltato")
                else:
                    new_files.append(file_path)
                    logger.info(f"Nuovo file da processare: {file_name}")
            except Exception as e:
                logger.warning(f"Errore nel controllo del file {file_path}: {str(e)}. Considerato come nuovo.")
                new_files.append(file_path)
        
        # Informa l'utente
        if skipped_files and progress_callback:
            progress_callback(f"⏩ Saltati {len(skipped_files)} file già vettorizzati: {', '.join(skipped_files[:3])}" + 
                            (f" e altri {len(skipped_files)-3}..." if len(skipped_files) > 3 else ""))
        
        if not new_files:
            if progress_callback:
                progress_callback("✅ Tutti i file sono già stati vettorizzati!")
            logger.info("Tutti i file sono già stati vettorizzati")
            
            # Carica il vector store esistente
            if global_vs_exists:
                try:
                    vector_store = FAISS.load_local(
                        global_vs_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    self.vector_store = vector_store
                    return vector_store
                except Exception as e:
                    logger.error(f"Errore nel caricamento del vector store globale: {str(e)}")
                    raise
            else:
                raise ValueError("Nessun file nuovo da processare e nessun vector store globale esistente")
        
        # Informa su quanti file nuovi verranno processati
        if progress_callback:
            progress_callback(f"🔄 Processamento di {len(new_files)} nuovi file...")
        logger.info(f"Processamento di {len(new_files)} nuovi file")
        
        # Processa i nuovi file
        all_chunks = []
        new_processed_files = {}
        
        for i, file_path in enumerate(new_files):
            try:
                if progress_callback:
                    progress_callback(f"Convertendo {i+1}/{len(new_files)}: {os.path.basename(file_path)}")
                
                # Carica PDF come Markdown
                if split_by_pages:
                    documents = self.load_pdf_as_markdown_pages(file_path)
                else:
                    documents = self.load_pdf_as_markdown(file_path)
                
                # Processa i documenti in chunks
                chunks = self.process_documents(documents)
                all_chunks.extend(chunks)
                
                # Aggiungi il file ai processati con il suo hash
                file_hash = self._get_file_hash(file_path)
                file_name = os.path.basename(file_path)
                new_processed_files[file_hash] = {
                    'path': file_path,
                    'filename': file_name,
                    'chunks_count': len(chunks),
                    'processed_at': time.time()
                }
                
                logger.info(f"File {i+1}/{len(new_files)} convertito: {len(chunks)} chunks da {file_name}")
                
            except Exception as e:
                logger.error(f"Errore nella conversione di {file_path}: {str(e)}")
                if progress_callback:
                    progress_callback(f"⚠️ Errore nel file {os.path.basename(file_path)}: {str(e)}")
                continue
        
        if not all_chunks:
            if not global_vs_exists:
                raise ValueError("Nessun documento convertito con successo e nessun vector store esistente")
            elif progress_callback:
                progress_callback("⚠️ Nessun nuovo documento convertito con successo, uso vector store esistente")
            
            # Carica il vector store esistente
            try:
                vector_store = FAISS.load_local(
                    global_vs_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.vector_store = vector_store
                return vector_store
            except Exception as e:
                logger.error(f"Errore nel caricamento del vector store globale: {str(e)}")
                raise
        
        # Fase 2: Creazione o aggiornamento del vector store
        if global_vs_exists:
            # Carica il vector store esistente
            if progress_callback:
                progress_callback(f"Caricamento vector store esistente...")
            
            try:
                vector_store = FAISS.load_local(
                    global_vs_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Aggiungi i nuovi chunks
                if progress_callback:
                    progress_callback(f"Aggiunta di {len(all_chunks)} nuovi chunks al vector store...")
                
                # Crea un vector store temporaneo con i nuovi chunks
                temp_vs = self.create_vector_store_batch(all_chunks, progress_callback)
                
                # Combina con il vector store esistente
                vector_store.merge_from(temp_vs)
                
                logger.info(f"Aggiunti {len(all_chunks)} nuovi chunks al vector store globale")
                
            except Exception as e:
                logger.error(f"Errore nell'aggiornamento del vector store globale: {str(e)}")
                if progress_callback:
                    progress_callback(f"⚠️ Errore nell'aggiornamento, creazione nuovo vector store...")
                
                # Se c'è un errore, crea un nuovo vector store
                vector_store = self.create_vector_store_batch(all_chunks, progress_callback)
        else:
            # Crea un nuovo vector store
            if progress_callback:
                progress_callback(f"Creazione nuovo vector store con {len(all_chunks)} chunks...")
            
            vector_store = self.create_vector_store_batch(all_chunks, progress_callback)
        
        # Aggiorna i metadati globali
        processed_files.update(new_processed_files)
        
        # Crea un indice per nome file per facilitare le ricerche future
        filename_index = {}
        for file_hash, file_info in processed_files.items():
            if 'filename' in file_info:
                filename_index[file_info['filename']] = file_hash
        
        global_metadata = {
            'processed_files': processed_files,
            'filename_index': filename_index,
            'last_updated': time.time(),
            'total_files': len(processed_files),
            'total_chunks': sum(f.get('chunks_count', 0) for f in processed_files.values()),
            'config': {
                'chunk_size': self.config.CHUNK_SIZE,
                'chunk_overlap': self.config.CHUNK_OVERLAP,
                'batch_size': self.config.BATCH_SIZE
            }
        }
        
        # Salva il vector store aggiornato
        try:
            if progress_callback:
                progress_callback("Salvataggio vector store globale...")
            
            # Salva il vector store
            vector_store.save_local(global_vs_path)
            
            # Salva i metadati
            with open(global_meta_path, 'w', encoding='utf-8') as f:
                json.dump(global_metadata, f, indent=2, ensure_ascii=False)
            
            if progress_callback:
                progress_callback(f"✅ Vector store globale aggiornato con {len(processed_files)} file totali!")
            
            logger.info(f"Vector store globale salvato: {global_vs_path}")
            logger.info(f"Totale file: {len(processed_files)}, Totale chunks: {global_metadata['total_chunks']}")
            
        except Exception as e:
            logger.error(f"Errore nel salvataggio del vector store globale: {str(e)}")
            if progress_callback:
                progress_callback(f"⚠️ Errore nel salvataggio: {str(e)}")
        
        self.vector_store = vector_store
        return vector_store
    
    def _get_files_hash(self, file_paths: List[str]) -> str:
        """Genera un hash unico basato sui contenuti dei file"""
        hasher = hashlib.sha256()
       
        # Ordina i percorsi per garantire un hash consistente
        sorted_paths = sorted(file_paths)
       
        for file_path in sorted_paths:
            try:
                # Aggiungi il percorso del file all'hash
                hasher.update(file_path.encode('utf-8'))
               
                # Aggiungi il contenuto del file all'hash
                with open(file_path, 'rb') as f:
                    # Leggi il file a blocchi per gestire file grandi
                    for chunk in iter(lambda: f.read(4096), b''):
                        hasher.update(chunk)
                       
                # Aggiungi la data di modifica del file all'hash
                mtime = str(os.path.getmtime(file_path))
                hasher.update(mtime.encode('utf-8'))
               
            except Exception as e:
                logger.error(f"Errore nel calcolo hash per {file_path}: {str(e)}")
                continue
       
        # Aggiungi anche i parametri di configurazione per rendere l'hash dipendente dalla configurazione
        config_data = json.dumps({
            'chunk_size': self.config.CHUNK_SIZE,
            'chunk_overlap': self.config.CHUNK_OVERLAP,
            'batch_size': self.config.BATCH_SIZE
        }, sort_keys=True)
        hasher.update(config_data.encode('utf-8'))
       
        return hasher.hexdigest()
    
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
    
    def _get_file_hash(self, file_path: str) -> str:
        """Genera un hash unico basato sul contenuto e data modifica di un singolo file"""
        hasher = hashlib.sha256()
        
        try:
            # Aggiungi il percorso del file all'hash
            hasher.update(file_path.encode('utf-8'))
            
            # Aggiungi il contenuto del file all'hash
            with open(file_path, 'rb') as f:
                # Leggi il file a blocchi per gestire file grandi
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
                    
            # Aggiungi la data di modifica del file all'hash
            mtime = str(os.path.getmtime(file_path))
            hasher.update(mtime.encode('utf-8'))
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Errore nel calcolo hash per {file_path}: {str(e)}")
            raise
