# Sistema RAG per Analisi Bandi Pubblici

Un sistema RAG (Retrieval Augmented Generation) per l'analisi intelligente di bandi pubblici, che permette di:
- Caricare e processare documenti PDF di bandi
- Fare domande in linguaggio naturale sui bandi
- Ottenere risposte precise con riferimenti alle fonti
- Generare tabelle di sintesi e documenti riassuntivi

## Architettura del Sistema

```mermaid
graph TB
    subgraph Frontend
        UI[Streamlit UI]
        Upload[Upload Manager]
        Chat[Chat Interface]
    end
    
    subgraph Backend
        DP[Document Processor]
        VS[Vector Store]
        RAG[RAG System]
        CM[Chat Manager]
    end
    
    subgraph External
        AOAI[Azure OpenAI]
        FS[File System]
    end
    
    UI --> Upload
    UI --> Chat
    Upload --> DP
    DP --> VS
    Chat --> RAG
    RAG --> VS
    RAG --> AOAI
    CM --> FS
    RAG --> CM
```

## Componenti Principali

```mermaid
classDiagram
    class DocumentProcessor {
        +process_multiple_files()
        +load_pdf()
        +create_vector_store()
    }
    
    class RAGSystem {
        +setup_qa_chain()
        +query()
        +analyze_user_intent()
        -ConversationBufferMemory memory
    }
    
    class ChatManager {
        +create_session()
        +add_message()
        +save_session()
        +load_sessions()
    }
    
    class ChatSession {
        +session_id: str
        +messages: List
        +add_message()
        +to_dict()
    }
    
    RAGSystem --> ChatManager
    ChatManager --> ChatSession
    RAGSystem --> DocumentProcessor
```

## Workflow di una Query

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Streamlit UI
    participant R as RAG System
    participant CM as Chat Manager
    participant VS as Vector Store
    participant LLM as Azure OpenAI

    U->>UI: Invia domanda
    UI->>R: query(question)
    R->>R: analyze_user_intent()
    
    alt È una domanda sui bandi
        R->>VS: search_documents()
        VS-->>R: relevant_chunks
        R->>CM: get_chat_history()
        CM-->>R: formatted_history
        R->>LLM: generate_response(chunks + history)
        LLM-->>R: response + sources
    else È chat generale
        R->>CM: get_chat_history()
        CM-->>R: formatted_history
        R->>LLM: generate_chat_response(history)
        LLM-->>R: response
    end
    
    R->>CM: save_message()
    R-->>UI: response
    UI-->>U: Mostra risposta
```

## Sistema di Memoria

```mermaid
graph LR
    subgraph Memory System
        CM[Chat Manager]
        PS[Persistent Storage]
        LM[LangChain Memory]
    end
    
    subgraph Memory Types
        BM[Buffer Memory]
        WM[Window Memory]
        SM[Summary Memory]
        SBM[Summary Buffer Memory]
    end
    
    CM --> PS
    CM --> LM
    LM --> BM
    LM --> WM
    LM --> SM
    LM --> SBM
```

## Funzionalità Principali

1. **Gestione Documenti**
   - Caricamento multiplo di PDF
   - Processamento e chunking automatico
   - Creazione embeddings con Azure OpenAI
   - Storage vettoriale per ricerca semantica

2. **Sistema RAG**
   - Analisi semantica dell'intento utente
   - Retrieval contestuale dei documenti
   - Generazione risposte con citazione fonti
   - Memoria conversazionale intelligente

3. **Gestione Chat**
   - Sessioni multiple
   - Persistenza su file system
   - Memoria conversazionale
   - Esportazione chat in vari formati

4. **Funzionalità Avanzate**
   - Tabella di sintesi dei bandi
   - Ricerca per idea progettuale
   - Documento di sintesi automatico
   - Analisi semantica delle domande

## Setup e Configurazione

1. Installare le dipendenze:
```bash
pip install -r requirements.txt
```

2. Configurare le variabili d'ambiente in `.env`:
```env
AZURE_API_KEY=your_key
AZURE_ENDPOINT=your_endpoint
AZURE_API_VERSION=2024-02-01
DEPLOYMENT_NAME=your_deployment
```

3. Avviare l'applicazione:
```bash
streamlit run app.py
```

## Struttura Directory

```
project/
├── src/
│   ├── __init__.py
│   ├── ai_processor.py
│   ├── chat_manager.py
│   ├── config.py
│   ├── document_processor.py
│   ├── rag_system.py
│   └── utils.py
├── data/
│   └── processed/
├── docs/
│   ├── ARCHITECTURE.md
│   └── API.md
├── tests/
│   └── test_*.py
├── app.py
├── requirements.txt
└── README.md
```