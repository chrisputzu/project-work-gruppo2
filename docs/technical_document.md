# Documento Tecnico - Bandi Assistant
## Sistema RAG per Analisi Bandi Pubblici della Regione Lombardia

### Versione 1.0
### Data: Luglio 2025

---

# Sommario 

[Architettura applicativa](#architettura-applicativa)  
[Business Architecture](#business-architecture)  
[Information System Architecture](#information-system-architecture)  
[Architettura Tecnica](#architettura-tecnica)  
[Application Technical Architecture](#application-technical-architecture)  
[Risorse Cloud](#risorse-cloud)  
[Release Management](#release-management)  
[Copertura Requisiti Utente](#copertura-requisiti-utente)  
[Assunzioni, vincoli e validità dell'offerta](#assunzioni-vincoli-e-validità-dellofferta)

---

# Architettura applicativa

Il progetto **Bandi Assistant** è un sistema RAG (Retrieval Augmented Generation) sviluppato da **EY**. L'obiettivo è fornire un assistente intelligente per l'analisi e la consultazione dei bandi pubblici, permettendo agli utenti di:

- Interrogare in linguaggio naturale i contenuti dei bandi
- Ottenere risposte precise con citazione delle fonti
- Generare tabelle di sintesi automatiche
- Identificare bandi compatibili con specifiche idee progettuali
- Esportare dati in formati strutturati (Excel, CSV, Markdown)

La soluzione implementa tecnologie all'avanguardia di **Natural Language Processing** e **Intelligenza Artificiale** per trasformare documenti PDF complessi in un sistema di consultazione intelligente e interattivo.

## Business Architecture

Il sistema Bandi Assistant si inserisce nel processo di **gestione e consultazione dei bandi pubblici** della Regione Lombardia, digitalizzando e automatizzando le attività di analisi documentale.

### Processi impattati:

| **Processo** | **Fase** | **Sistema Applicativo (AP) / Applicazione (PK)** | **Ruolo dell'Applicativo** | **Impatto Applicativo (S/N)** | **Applicativo esistente o nuovo** |
|--------------|----------|---------------------------------------------------|----------------------------|-------------------------------|-----------------------------------|
| Gestione Bandi Pubblici | Pubblicazione e Consultazione | Bandi Assistant (PK) | Sistema principale per consultazione intelligente | S | Nuovo |
| Analisi Documentale | Processamento PDF | Document Processor (AP) | Conversione PDF→Markdown e chunking | S | Nuovo |
| Supporto Utenti | Assistenza e Consulenza | Chat AI System (AP) | Risposta automatica a domande su bandi | S | Nuovo |
| Reporting e Analytics | Generazione Report | Export Manager (AP) | Generazione tabelle di sintesi e report | S | Nuovo |
| Archiviazione Documenti | Storage e Indicizzazione | Vector Store (AP) | Indicizzazione semantica documenti | S | Nuovo |

**Tabella 1: Business Architecture - dettaglio applicativi**

## Information System Architecture

Il sistema è strutturato in una architettura modulare che comprende:

1. **Frontend Layer**: Interfaccia utente Streamlit responsiva
2. **Processing Layer**: Motori di elaborazione documenti e AI
3. **Storage Layer**: Persistenza dati e vector store
4. **Integration Layer**: Integrazione con servizi Azure OpenAI

### Flussi dati principali:

| **Flusso dati** | **Descrizione** |
|-----------------|-----------------|
| PDF_UPLOAD | Caricamento documenti PDF dei bandi dalla cartella data/ o upload manuale |
| PDF_TO_MARKDOWN | Conversione documenti PDF in formato Markdown strutturato tramite pymupdf4llm |
| DOCUMENT_CHUNKING | Suddivisione documenti in chunk semantici con overlap configurabile |
| EMBEDDING_GENERATION | Generazione embeddings vettoriali tramite Azure OpenAI text-embedding-ada-002 |
| VECTOR_STORAGE | Persistenza embeddings in FAISS vector store con cache intelligente |
| SEMANTIC_SEARCH | Ricerca semantica nei documenti basata su query utente |
| LLM_PROCESSING | Elaborazione risposte tramite Azure OpenAI GPT-4o con context injection |
| CHAT_PERSISTENCE | Salvataggio conversazioni in formato JSON con metadati |
| EXPORT_DATA | Esportazione dati in formato Excel/CSV/Markdown |

**Tabella 2: Information Architecture - dettaglio flussi**

---

# Architettura Tecnica

## Application Technical Architecture

Il sistema implementa un'architettura a microservizi con i seguenti componenti principali:

### Stack Tecnologico:

| **Pacchetto Applicativo** | **Software** | **Versione** | **Layer Applicativo** | **Applicativo nuovo o esistente** |
|---------------------------|--------------|--------------|----------------------|-----------------------------------|
| Frontend Interface | Streamlit | 1.46.1 | Presentation | Nuovo |
| AI Processing Engine | Azure OpenAI GPT-4o | 2024-02-01 | Application | Esistente (servizio) |
| Embedding Service | Azure OpenAI Embeddings | text-embedding-ada-002 | Application | Esistente (servizio) |
| PDF Processing | PyMuPDF + pymupdf4llm | 1.26.3 / 0.0.26 | Application | Nuovo |
| Vector Database | FAISS | 1.8.0 | Data | Nuovo |
| Document Processing | LangChain | 0.3.26 | Application | Nuovo |
| Data Export | Pandas + openpyxl | 2.3.1 / 3.1.5 | Application | Nuovo |
| Session Management | Python JSON | 3.11+ | Data | Nuovo |
| Configuration | Python-dotenv | 1.1.1 | Application | Nuovo |
| Logging System | Python Logging | 3.11+ | Application | Nuovo |

**Tabella 3: Technical Architecture - dettaglio applicativi**

### Flussi Dati Tecnici:

| **Flusso dati** | **Protocollo** | **Origine Layer** | **Destinazione Layer** | **Tipologia** | **Frequenza** | **Frequenza di picco** | **Dimensione media dei messaggi** | **Dimensione massima dei messaggi** | **Flusso nuovo o esist.** |
|-----------------|----------------|------------------|------------------------|---------------|--------------|----------------------|----------------------------------|-----------------------------------|--------------------------|
| PDF Upload | HTTP POST | Presentation | Application | Real-time | Su richiesta | 10 file/min | 1MB | 50MB | Nuovo |
| Embedding API | HTTPS REST | Application | Azure OpenAI | Real-time | 50 req/min | 200 req/min | 2KB | 10KB | Nuovo |
| LLM API | HTTPS REST | Application | Azure OpenAI | Real-time | 10 req/min | 50 req/min | 15KB | 100KB | Nuovo |
| Vector Search | In-memory | Application | Data | Real-time | 100 req/min | 500 req/min | 1KB | 5KB | Nuovo |
| Chat Persistence | File I/O | Application | Data | Batch | 1 req/min | 10 req/min | 5KB | 50KB | Nuovo |
| Export Data | HTTP Response | Application | Presentation | On-demand | 1 req/ora | 5 req/ora | 100KB | 5MB | Nuovo |

**Tabella 4: Technical Architecture - flussi dati**

---

# Risorse Cloud

Il sistema utilizza **Azure Cloud** come provider principale per i servizi di AI e può essere deployato sia on-premise che in cloud.

| (AP/PK) | Ambiente | Risorsa | Tipologia | Cloud Provider | Region | Resource Group | FQDN Front Door | FQDN Backend | RAM | Storage | CPU | Core | SO | SW |
|---------|----------|---------|-----------|---------------|--------|----------------|----------------|--------------|-----|---------|-----|------|----|----|
| Bandi Assistant | Sviluppo | Azure OpenAI Service | PaaS | Microsoft Azure | West Europe | RG-BandiAI-Dev | bandiassistant-dev.azurewebsites.net | aoai-bandiassistant-dev.openai.azure.com | N/A | N/A | N/A | N/A | N/A | GPT-4o, text-embedding-ada-002 |
| Bandi Assistant | Collaudo | Azure OpenAI Service | PaaS | Microsoft Azure | West Europe | RG-BandiAI-Test | bandiassistant-test.azurewebsites.net | aoai-bandiassistant-test.openai.azure.com | N/A | N/A | N/A | N/A | N/A | GPT-4o, text-embedding-ada-002 |
| Bandi Assistant | Produzione | Azure OpenAI Service | PaaS | Microsoft Azure | West Europe | RG-BandiAI-Prod | bandiassistant.regione.lombardia.it | aoai-bandiassistant-prod.openai.azure.com | N/A | N/A | N/A | N/A | N/A | GPT-4o, text-embedding-ada-002 |
| App Server | Sviluppo | Azure Container Instance | PaaS | Microsoft Azure | West Europe | RG-BandiAI-Dev | N/A | N/A | 4GB | 20GB | 2 vCPU | 2 | Linux Ubuntu 22.04 | Python 3.11, Streamlit |
| App Server | Collaudo | Azure Container Instance | PaaS | Microsoft Azure | West Europe | RG-BandiAI-Test | N/A | N/A | 8GB | 50GB | 4 vCPU | 4 | Linux Ubuntu 22.04 | Python 3.11, Streamlit |
| App Server | Produzione | Azure Container Instance | PaaS | Microsoft Azure | West Europe | RG-BandiAI-Prod | N/A | N/A | 16GB | 100GB | 8 vCPU | 8 | Linux Ubuntu 22.04 | Python 3.11, Streamlit |

**Tabella 5: Risorse Cloud**

### Stima Costi Azure (mensili):
- **Sviluppo**: €200-300/mese
- **Collaudo**: €400-600/mese  
- **Produzione**: €800-1200/mese

*(Include: Azure OpenAI tokens, Container Instances, Storage, Networking)*

---

# Release Management

| Piattaforma | Nome release (se applicazione a release) | Ordine PIP e tipologia |
|-------------|------------------------------------------|----------------------|
| Bandi Assistant Core | v1.0-MVP | 1. BS (Business Service) - Rilascio iniziale con funzionalità core |
| Document Processor Enhancement | v1.1-DocProc | 2. BS - Miglioramenti conversione PDF-Markdown |
| Chat AI Advanced | v1.2-ChatAI | 3. BS - Funzionalità chat avanzate e memoria estesa |
| Export & Analytics | v1.3-Export | 4. BS - Modulo export e analytics avanzato |
| Performance Optimization | v1.4-Perf | 5. BS - Ottimizzazioni performance e scalabilità |

**Sequenza di rilascio:**
1. **MVP (v1.0)**: Core RAG system con chat basic
2. **Enhancement (v1.1)**: Miglioramento processamento documenti
3. **Advanced Features (v1.2)**: Chat intelligence e memoria avanzata
4. **Analytics (v1.3)**: Report e dashboard avanzati
5. **Production Ready (v1.4)**: Ottimizzazioni per produzione

---

# Copertura Requisiti Utente

| **Requisito Utente** | **Copertura** | **Soluzione Implementata** |
|----------------------|---------------|---------------------------|
| **RF01** - Caricamento documenti PDF |  Completa | Modulo upload con validazione file, supporto batch e cartella data/ |
| **RF02** - Conversione PDF in formato consultabile |  Completa | Conversione PDF→Markdown con pymupdf4llm, preservazione struttura |
| **RF03** - Ricerca semantica nei documenti |  Completa | Vector store FAISS con embeddings Azure OpenAI |
| **RF04** - Chat intelligente sui bandi |  Completa | Sistema RAG con GPT-4o, analisi intento utente, memoria conversazionale |
| **RF05** - Citazione fonti nelle risposte |  Completa | Automatic source citation con preview contenuto |
| **RF06** - Generazione tabelle di sintesi |  Completa | Estrazione automatica metadati bandi, tabella editabile |
| **RF07** - Export dati strutturati |  Completa | Export Excel/CSV con formattazione personalizzata |
| **RF08** - Gestione sessioni multiple |  Completa | Chat manager con persistenza JSON, cronologia conversazioni |
| **RF09** - Interfaccia user-friendly |  Completa | Streamlit responsive con logo EY |
| **RF10** - Ricerca per idea progettuale |  Completa | Modalità specializzata per matching bandi-progetti |
| **RNF01** - Performance < 3s per query |  Completa | Cache intelligente, batch processing, ottimizzazione FAISS |
| **RNF02** - Scalabilità 100+ documenti |  Completa | Vector store scalabile, processamento batch con retry |
| **RNF03** - Affidabilità 99.5% uptime |  Completa | Error handling robusto, fallback automatici |
| **RNF04** - Sicurezza dati sensibili |  Completa | Configurazione sicura Azure OpenAI, storage locale |

---

# Assunzioni, vincoli e validità dell'offerta

## Assunzioni

### Volume di utilizzo stimato:
- **Documenti PDF**: 50-200 bandi per deployment
- **Utenti concorrenti**: 10-50 utenti simultanei
- **Query giornaliere**: 500-2000 interrogazioni/giorno
- **Dimensione media PDF**: 5MB per documento
- **Sessioni chat**: 20-100 sessioni attive/giorno

### Assunzioni tecniche:
- **Disponibilità Azure OpenAI**: Servizio disponibile 99.9% del tempo
- **Formato documenti**: PDF testuali (non scansioni OCR)
- **Lingua**: Documenti in italiano standard
- **Connettività**: Connessione internet stabile per API Azure
- **Risorse hardware**: Minimo 8GB RAM, 4 CPU cores per produzione

## Vincoli

### Vincoli tecnici:
- **Rate limits Azure OpenAI**: 
  - GPT-4o: 300 requests/minute
  - Embeddings: 1500 requests/minute
- **Dimensione file**: Limite 50MB per singolo PDF
- **Lingue supportate**: Italiano (ottimizzato), inglese 
- **Formati input**: Solo PDF (no Word, Excel, PowerPoint)

### Vincoli operativi:
- **Deployment**: Richiede configurazione chiavi Azure OpenAI
- **Manutenzione**: Pulizia periodica cache (settimanale)
- **Backup**: Backup manuale sessioni chat e configurazioni

### Vincoli di sicurezza:
- **Dati sensibili**: Storage locale, no cloud pubblico per documenti
- **Accesso**: No autenticazione multi-user (single instance)
- **Compliance**: GDPR compliance per chat history

## Validità dell'offerta

### Validità temporale:
- **Durata**: 12 mesi dall'accettazione
- **Supporto**: 6 mesi di supporto tecnico incluso
- **Aggiornamenti**: Aggiornamenti minori inclusi per 6 mesi

### Condizioni di validità:
- **Configurazione Azure**: Cliente fornisce credenziali Azure OpenAI
- **Ambiente deployment**: Ambiente Linux/Windows compatibile
- **Documenti test**: Cliente fornisce set documenti per testing
- **Acceptance**: Testing e acceptance entro 30 giorni

---

**Documento preparato da:** 
- Alessandro Alfieri
- Palmina Angelini
- Joanna Ben Kakitie
- Christian Putzu
- Nicola Stara

**Data:** Luglio 2025  
**Versione:** 1.0  
**Stato:** In attesa di approvazione per rilascio MVP