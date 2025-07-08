import os
from dotenv import load_dotenv

# Carica variabili d'ambiente
load_dotenv()

class Config:
    """Configurazione del sistema"""
    # Azure OpenAI
    AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT")
    AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
    
    # Azure OpenAI Embeddings (dedicato)
    AZURE_EMBEDDING_ENDPOINT = os.environ.get("AZURE_ENDPOINT_EMB")
    AZURE_EMBEDDING_API_KEY = os.environ.get("AZURE_API_KEY_EMB")
    
    # Configurazione Azure
    AZURE_API_VERSION = "2024-02-01"
    DEPLOYMENT_NAME = "gpt-4o"
    AZURE_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"
    
    # Configurazione generale
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_TOKENS = 4000
    TEMPERATURE = 0.1
    
    # Configurazione batch processing
    BATCH_SIZE = 50  # Numero di chunks per batch
    BATCH_DELAY = 2  # Secondi di attesa tra batch
    MAX_RETRIES = 3  # Numero massimo di retry
    RETRY_DELAY = 5  # Secondi di attesa iniziale per retry
    
    # Percorsi
    DATA_DIR = "data"
    VECTOR_STORE_DIR = "vector_store"
    UPLOADS_DIR = "uploads"

    @classmethod
    def use_azure_openai(cls):
        """Verifica se usare Azure OpenAI dedicato"""
        return bool(cls.AZURE_API_KEY and cls.AZURE_ENDPOINT)
    
    @classmethod
    def use_azure_embeddings(cls):
        """Verifica se usare Azure Embeddings dedicato"""
        return bool(cls.AZURE_EMBEDDING_API_KEY and cls.AZURE_EMBEDDING_ENDPOINT)

    @classmethod
    def validate_config(cls):
        """Valida la configurazione"""
        # Verifica configurazione base Azure
        if not cls.use_azure_openai():
            raise ValueError("Configurazione LLM Azure API mancante. Fornire AZURE_API_KEY e AZURE_ENDPOINT")
        
        # Verifica configurazione embeddings
        if not cls.use_azure_embeddings():
            raise ValueError("Configurazione Embeddings Azure API mancante. Fornire AZURE_API_KEY_EMB e AZURE_ENDPOINT_EMB o configurazione Azure standard")
        
        return True

