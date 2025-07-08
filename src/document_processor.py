import os
import logging

from typing import List, Dict, Any
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