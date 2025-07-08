import streamlit as st
import os
import pandas as pd
from typing import List, Dict, Any
import logging
from pathlib import Path
from datetime import datetime

# Import dei moduli locali
from src.config import Config
from src.document_processor import EnhancedDocumentProcessor
from src.rag_system import RAGSystem
from src.chat_manager import ChatManager
from src.utils import (
    setup_logging, create_directories, save_uploaded_file, 
    export_to_excel, export_to_csv, validate_pdf_file,
    format_file_size, save_session_state, load_session_state,
    clear_session_state, get_file_stats
)

# Configurazione della pagina
st.set_page_config(
    page_title="Bandi Assistant",
    page_icon="logo/logo_lombardIA.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup iniziale
setup_logging()
create_directories()

# Configurazione del logger
logger = logging.getLogger(__name__)

class BandiRAGApp:
    """Applicazione principale per il sistema RAG dei bandi"""
    
    def __init__(self):
        self.config = Config()
        # Usa il processore migliorato con conversione Markdown
        self.document_processor = EnhancedDocumentProcessor()
        self.rag_system = RAGSystem()
        self.vector_store = None
        self.chat_manager = ChatManager()
        
        # Inizializza lo stato della sessione
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []
        if 'vector_store_ready' not in st.session_state:
            st.session_state.vector_store_ready = False
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'chat_session_id' not in st.session_state:
            st.session_state.chat_session_id = None
        if 'suggested_query' not in st.session_state:
            st.session_state.suggested_query = None
        
        # Aggiungi opzioni per la conversione Markdown
        if 'use_markdown_conversion' not in st.session_state:
            st.session_state.use_markdown_conversion = True
        if 'split_by_pages' not in st.session_state:
            st.session_state.split_by_pages = False
    
    def validate_configuration(self):
        """Valida la configurazione dell'applicazione"""
        try:
            self.config.validate_config()
            return True
        except ValueError as e:
            st.error(f"❌ Errore di configurazione: {str(e)}")
            st.info("💡 Assicurati di aver configurato correttamente le chiavi API nel file .env")
            return False
    
    def render_sidebar(self):
        """Renderizza la sidebar con le opzioni di navigazione"""
        # Logo in sidebar centrato usando le colonne di Streamlit
        logo_path = Path("logo/logo_lombardIA.png")
        if logo_path.exists():
            col1, col2 = st.sidebar.columns([1, 3])
            with col1:
                st.image(str(logo_path), width=50)
            with col2:
                st.markdown(
                    """
                    <div style='display: flex; align-items: flex-end; height: 50px;'>
                        <span style='font-size: 1.3em; font-weight: bold; margin-left: 0.2em; margin-bottom: 0;'>Bandi Assistant</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            
        # Menu di navigazione
        page = st.sidebar.selectbox(
            "Seleziona una funzione:",
            [
                "📁 Gestione Documenti",
                "💬 Chat con AI", 
                "📊 Tabella Riassuntiva",
                "📄 Report di Sintesi"
            ]
        )
        
        # Statistiche semplificate
        if st.session_state.processed_files:
            st.sidebar.markdown("---")
            st.sidebar.metric("Documenti caricati", len(st.session_state.processed_files))
            
            # Mostra info conversione se disponibile
            if st.session_state.documents:
                markdown_docs = sum(1 for doc in st.session_state.documents 
                                  if doc.metadata.get('content_type') == 'markdown')
                if markdown_docs > 0:
                    st.sidebar.metric("Convertiti in Markdown", markdown_docs)
            
            # Gestione chat
            sessions = self.chat_manager.get_session_list()
            if sessions:
                st.sidebar.markdown("---")
                
            
                
                
                # Pulsante per nuova chat
                if st.sidebar.button("➕ Nuova Chat", key="new_chat_btn"):
                    metadata = {
                        "created_by": "user",
                        "documents_count": len(st.session_state.processed_files),
                        "documents": [os.path.basename(f) for f in st.session_state.processed_files],
                        "conversion_method": "markdown" if st.session_state.use_markdown_conversion else "standard"
                    }
                    session_id = self.chat_manager.create_session(metadata)
                    st.session_state.chat_session_id = session_id
                    st.rerun()

                # Menu chat
                st.sidebar.markdown("### Cronologia chat")
                
                # Lista chat esistenti
                for session in sessions[:5]:  # Mostra solo le prime 5
                    summary = self.chat_manager.get_session_summary(session['session_id'])
                    if summary:
                        chat_title = f"💭 {datetime.fromisoformat(session['created_at']).strftime('%d/%m %H:%M')} ({summary['message_count']} msg)"
                    else:
                        chat_title = f"💭 {datetime.fromisoformat(session['created_at']).strftime('%d/%m %H:%M')}"
                    
                    if st.sidebar.button(chat_title, key=f"chat_{session['session_id']}"):
                        st.session_state.chat_session_id = session['session_id']
                        self.chat_manager.set_current_session(session['session_id'])
                        st.rerun()
        
        # Pulsante per resettare
        st.sidebar.markdown("---")
        if st.sidebar.button("🔄 Reset Sistema"):
            self.reset_system()
        
        return page
    
    def reset_system(self):
        """Resetta il sistema pulendo tutti i dati"""
        st.session_state.processed_files = []
        st.session_state.vector_store_ready = False
        st.session_state.documents = []
        st.session_state.vector_store = None
        st.session_state.chat_session_id = None
        clear_session_state()
        st.success("✅ Sistema resettato!")
        st.rerun()
    
    def process_data_folder(self):
        """Carica tutti i file PDF dalla cartella data"""
        data_path = Path(self.config.DATA_DIR)
        
        if not data_path.exists():
            st.error(f"❌ La cartella {self.config.DATA_DIR} non esiste!")
            return []
        
        pdf_files = list(data_path.glob("*.pdf"))
        
        if not pdf_files:
            st.warning(f"⚠️ Nessun file PDF trovato nella cartella")
            return []
        
        # Filtra i file già processati
        already_processed = set()
        for processed_file in st.session_state.processed_files:
            try:
                normalized_path = Path(processed_file).resolve()
                already_processed.add(str(normalized_path))
            except:
                already_processed.add(os.path.basename(processed_file))
        
        new_files = []
        for pdf_file in pdf_files:
            normalized_pdf = str(pdf_file.resolve())
            if normalized_pdf not in already_processed:
                if os.path.basename(str(pdf_file)) not in already_processed:
                    new_files.append(pdf_file)
        
        if not new_files:
            st.info("ℹ️ Tutti i file nella cartella data sono già stati processati")
            return []
        
        return new_files
    
    def render_file_upload_page(self):
        """Pagina per il caricamento dei documenti con opzioni Markdown"""
        st.title("📁 Gestione Documenti")
        
        st.markdown("""
        Carica i documenti PDF dei bandi pubblici della Regione Lombardia. 
        """)
        
        
        
        # Sezione per processare la cartella data
        st.markdown("### 📄 Elabora Documenti")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            available_files = self.process_data_folder()
            if available_files:
                
                # Mostra lista file
                with st.expander("📋 File da processare"):
                    for file in available_files:
                        file_stats = get_file_stats(str(file))
                        st.write(f"📄 {file.name} - {file_stats.get('size_formatted', 'N/A')}")
                st.info(f"📁 Trovati {len(available_files)} nuovi file PDF nella cartella")
            else:
                st.info(f"📁 Nessun nuovo file PDF da processare nella cartella '{self.config.DATA_DIR}'")
        
        with col2:
            if st.button("🚀 Processa Tutti", type="primary", disabled=not available_files):
                self.process_files_from_data_folder(available_files)
        
        st.markdown("---")
        
        # Sezione per upload manuale
        st.markdown("### 📤 Caricamento Manuale")
        uploaded_files = st.file_uploader(
            "Seleziona i file PDF dei bandi",
            type=['pdf'],
            accept_multiple_files=True,
            help="Carica uno o più file PDF contenenti i bandi pubblici"
        )
        
        if uploaded_files:
            
            total_size = 0
            valid_count = 0
            
            for file in uploaded_files:
                if validate_pdf_file(file):
                    
                    total_size += file.size
                    valid_count += 1
                else:
                    st.error(f"❌ {file.name} - File non valido")
            
            if valid_count > 0:
                st.info(f"{valid_count} file caricati ({format_file_size(total_size)})")
            
            if st.button("🚀 Processa Documenti", type="primary", disabled=valid_count == 0):
                self.process_uploaded_files(uploaded_files)
        
        
    
    def render_document_info_section(self):
        """Sezione per mostrare informazioni sui documenti processati"""
        st.markdown("---")
        st.markdown("### 📊 Informazioni Documenti Processati")
        
        # Statistiche generali
        total_docs = len(st.session_state.documents)
        markdown_docs = sum(1 for doc in st.session_state.documents 
                          if doc.metadata.get('content_type') == 'markdown')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documenti totali", total_docs)
        with col2:
            st.metric("Convertiti in Markdown", markdown_docs)
        with col3:
            st.metric("Metodo standard", total_docs - markdown_docs)
        with col4:
            if total_docs > 0:
                markdown_percentage = (markdown_docs / total_docs) * 100
                st.metric("% Markdown", f"{markdown_percentage:.1f}%")
        
        # Dettagli per file
        if st.expander("📋 Dettagli documenti"):
            try:
                doc_info = self.document_processor.extract_document_info(st.session_state.documents)
                if doc_info:
                    df = pd.DataFrame(doc_info)
                    # Riordina colonne per migliore visualizzazione
                    column_order = ['source', 'content_type', 'conversion_method', 'page', 'content_length', 'content_preview']
                    available_columns = [col for col in column_order if col in df.columns]
                    df_display = df[available_columns]
                    st.dataframe(df_display, use_container_width=True)
                else:
                    st.info("Nessuna informazione di dettaglio disponibile")
            except Exception as e:
                st.error(f"Errore nel mostrare dettagli: {str(e)}")
        
        # Debug conversione Markdown (se disponibile)
        if hasattr(self.rag_system, 'debug_markdown_processing') and markdown_docs > 0:
            if st.expander("🔍 Debug Conversione Markdown"):
                try:
                    debug_report = self.rag_system.debug_markdown_processing(st.session_state.documents)
                    st.markdown(debug_report)
                except Exception as e:
                    st.error(f"Errore nel debug: {str(e)}")
    
    def process_uploaded_files(self, uploaded_files):
        """Processa i file caricati manualmente con supporto Markdown"""
        valid_files = [f for f in uploaded_files if validate_pdf_file(f)]
        
        if not valid_files:
            st.error("❌ Nessun file PDF valido trovato!")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(message):
            status_text.text(message)
            logger.info(f"Progress: {message}")
        
        try:
            # Salva i file
            file_paths = []
            for i, file in enumerate(valid_files):
                status_text.text(f"Salvando {file.name}...")
                file_path = save_uploaded_file(file)
                file_paths.append(file_path)
                progress_bar.progress((i + 1) / (len(valid_files) * 4))
            
            # Processa i documenti con o senza conversione Markdown
            status_text.text("Processando documenti...")
            progress_bar.progress(0.25)
            
            if st.session_state.use_markdown_conversion:
                # Usa conversione Markdown
                self.vector_store = self.document_processor.process_multiple_files_markdown(
                    file_paths,
                    split_by_pages=st.session_state.split_by_pages,
                    progress_callback=update_progress
                )
            else:
                # Usa processamento standard
                self.vector_store = self.document_processor.process_multiple_files_batch(
                    file_paths,
                    progress_callback=update_progress
                )
            
            progress_bar.progress(0.75)
            
            # Configurazione del sistema RAG
            status_text.text("Configurando sistema RAG...")
            
            if not st.session_state.chat_session_id:
                metadata = {
                    "created_by": "user",
                    "documents_count": len(file_paths),
                    "documents": [os.path.basename(f) for f in file_paths],
                    "conversion_method": "markdown" if st.session_state.use_markdown_conversion else "standard"
                }
                session_id = self.chat_manager.create_session(metadata)
                st.session_state.chat_session_id = session_id
            
            # Usa sempre setup_qa_chain invece di setup_qa_chain_enhanced
            self.rag_system.setup_qa_chain(self.vector_store, st.session_state.chat_session_id)
            
            # Salva lo stato
            st.session_state.processed_files.extend(file_paths)
            st.session_state.vector_store_ready = True
            st.session_state.vector_store = self.vector_store
            
            # Carica documenti
            all_documents = []
            for file_path in file_paths:
                if st.session_state.use_markdown_conversion:
                    if st.session_state.split_by_pages:
                        docs = self.document_processor.load_pdf_as_markdown_pages(file_path)
                    else:
                        docs = self.document_processor.load_pdf_as_markdown(file_path)
                
                all_documents.extend(docs)
            
            st.session_state.documents.extend(all_documents)
            
            progress_bar.progress(1.0)
            conversion_method = "Markdown" if st.session_state.use_markdown_conversion else "Standard"
            status_text.text(f"✅ Processamento {conversion_method} completato!")
            st.success(f"🎉 Processati {len(valid_files)} documenti con conversione {conversion_method}!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Errore durante il processamento: {str(e)}")
            logger.error(f"Errore nel processamento: {str(e)}")
            
            if "429" in str(e) or "rate limit" in str(e).lower():
                st.warning("💡 **Rate limit raggiunto**: Il sistema batch includerà retry automatici. Attendi...")
            raise
    
    def process_files_from_data_folder(self, files_to_process):
        """Processa i file dalla cartella data con supporto Markdown"""
        if not files_to_process:
            st.warning("⚠️ Nessun file da processare nella cartella data.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(message):
            status_text.text(message)
            logger.info(f"Progress: {message}")
        
        try:
            file_paths = []
            for i, file_path in enumerate(files_to_process):
                status_text.text(f"Preparando {file_path.name}...")
                file_paths.append(str(file_path))
                progress_bar.progress((i + 1) / (len(files_to_process) * 4))
            
            status_text.text("Processando documenti...")
            progress_bar.progress(0.25)
            
            if st.session_state.use_markdown_conversion:
                # Usa conversione Markdown
                self.vector_store = self.document_processor.process_multiple_files_markdown(
                    file_paths,
                    split_by_pages=st.session_state.split_by_pages,
                    progress_callback=update_progress
                )
            else:
                # Usa processamento standard
                self.vector_store = self.document_processor.process_multiple_files_batch(
                    file_paths, 
                    progress_callback=update_progress
                )
            
            progress_bar.progress(0.75)
            
            status_text.text("Configurando sistema RAG...")
            
            if not st.session_state.chat_session_id:
                metadata = {
                    "created_by": "user",
                    "documents_count": len(file_paths),
                    "documents": [os.path.basename(f) for f in file_paths],
                    "conversion_method": "markdown" if st.session_state.use_markdown_conversion else "standard"
                }
                session_id = self.chat_manager.create_session(metadata)
                st.session_state.chat_session_id = session_id
            
            # Usa sempre setup_qa_chain invece di setup_qa_chain_enhanced
            self.rag_system.setup_qa_chain(self.vector_store, st.session_state.chat_session_id)
            
            # Salva lo stato
            st.session_state.processed_files.extend(file_paths)
            st.session_state.vector_store_ready = True
            st.session_state.vector_store = self.vector_store
            
            # Carica documenti
            status_text.text("Caricando documenti...")
            all_documents = []
            for file_path in file_paths:
                if st.session_state.use_markdown_conversion:
                    if st.session_state.split_by_pages:
                        docs = self.document_processor.load_pdf_as_markdown_pages(file_path)
                    else:
                        docs = self.document_processor.load_pdf_as_markdown(file_path)
                
                
                all_documents.extend(docs)
            
            st.session_state.documents.extend(all_documents)
            
            progress_bar.progress(1.0)
            conversion_method = "Markdown" if st.session_state.use_markdown_conversion else "Standard"
            status_text.text(f"✅ Processamento {conversion_method} completato!")
            st.success(f"🎉 Processati {len(files_to_process)} documenti dalla cartella data con conversione {conversion_method}!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Errore durante il processamento: {str(e)}")
            logger.error(f"Errore nel processamento: {str(e)}")
            
            if "429" in str(e) or "rate limit" in str(e).lower():
                st.warning("💡 **Suggerimento**: Se l'errore persiste, prova a:")
                st.markdown("- Attendere qualche minuto prima di riprovare")
            raise
    
    def render_chatbot_page(self):
        """Pagina del chatbot per domande sui bandi"""
        # Header con logo
        
        st.title("💬 Chat con AI")
        
        if not st.session_state.vector_store_ready:
            st.warning("⚠️ Carica prima alcuni documenti nella sezione 'Caricamento Documenti'")
            return
        
        # Gestione della sessione di chat
        if not st.session_state.chat_session_id:
            metadata = {
                "created_by": "user",
                "documents_count": len(st.session_state.processed_files),
                "documents": [os.path.basename(f) for f in st.session_state.processed_files],
                "conversion_method": "markdown" if st.session_state.use_markdown_conversion else "standard"
            }
            session_id = self.chat_manager.create_session(metadata)
            st.session_state.chat_session_id = session_id
        
        session = self.chat_manager.get_session(st.session_state.chat_session_id)
        if not session:
            st.error("❌ Errore nel caricamento della sessione")
            return
        
        # Controlla se c'è un suggerimento da processare
        if 'suggested_query' in st.session_state and st.session_state.suggested_query:
            suggestion = st.session_state.suggested_query
            # Aggiungi il messaggio alla chat
            session.add_message("user", suggestion)
            self.chat_manager.save_session(session)
            
            # Mostra il messaggio dell'utente
            with st.chat_message("user"):
                st.markdown(suggestion)
            
            # Elabora la risposta per il suggerimento
            with st.chat_message("assistant"):
                with st.spinner(f"Elaborando risposta per '{suggestion}'..."):
                    try:
                        if not hasattr(self, 'vector_store') or self.vector_store is None:
                            if st.session_state.vector_store is not None:
                                self.vector_store = st.session_state.vector_store
                                # Inizializza correttamente il RAG system prima di fare query
                                chat_history = "\n".join([
                                    f"{msg['role'].upper()}: {msg['content']}"
                                    for msg in session.messages
                                ])
                                self.rag_system.setup_qa_chain(
                                    self.vector_store, 
                                    session.session_id,
                                    initial_chat_history=chat_history
                                )
                            else:
                                st.error("❌ Vector store non disponibile.")
                                return
                                
                        result = self.rag_system.query(suggestion, session.session_id)
                        
                        # Aggiungi la risposta alla chat
                        session.add_message("assistant", result["answer"], result["sources"])
                        self.chat_manager.save_session(session)
                        
                        # Pulisci il suggerimento dopo averlo processato
                        st.session_state.suggested_query = None
                        st.rerun()
                    except Exception as e:
                        error_msg = f"❌ Errore nell'elaborazione del suggerimento: {str(e)}"
                        st.error(error_msg)
                        session.add_message("assistant", error_msg)
                        self.chat_manager.save_session(session)
                        st.session_state.suggested_query = None
                        st.rerun()
        
        # Info sessione
        st.caption(f"Chat iniziata il {datetime.fromisoformat(session.created_at).strftime('%d/%m/%Y alle %H:%M')}")
        
        
        

        # Suggerimenti intelligenti (se supportati)
        if st.expander("🔍 Suggerimenti di ricerca"):
            # Esempi di suggerimenti sempre disponibili
            example_suggestions = [
                "Quali sono i requisiti per partecipare al bando?",
                "Qual è la scadenza per presentare la domanda?",
                "Quali sono i settori finanziabili?",
                "Qual è il budget massimo per progetto?",
                "Chi sono i beneficiari ammissibili?"
            ]
            
            st.markdown("#### Esempi di domande:")
            col1, col2 = st.columns(2)
            for i, suggestion in enumerate(example_suggestions):
                with col1 if i % 2 == 0 else col2:
                    if st.button(suggestion, key=f"example_{hash(suggestion)}"):
                        st.session_state.suggested_query = suggestion
                        st.rerun()
            
            # Spazio vuoto intenzionale per mantenere solo i suggerimenti esempio
            
        # Chat container
        chat_container = st.container()
        with chat_container:
            for message in session.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    if "sources" in message:
                        with st.expander("📚 Fonti"):
                            for source in message["sources"]:
                                st.markdown(f"**{source['source']}**")
                                st.markdown(f"_{source['content_preview']}_")
        
        # Input personalizzato
        placeholder_text = {
            "general": "Fai una domanda sui bandi o parliamo...",
            "project_search": "Descrivi la tua idea progettuale...",
            "analysis": "Chiedi un'analisi o confronto..."
        }
        
        if prompt := st.chat_input("Scrivi qui la tua domanda o idea progettuale..."):
            session.add_message("user", prompt)
            self.chat_manager.save_session(session)
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Elaborando risposta..."):
                    try:
                        if not hasattr(self, 'vector_store') or self.vector_store is None:
                            if st.session_state.vector_store is not None:
                                self.vector_store = st.session_state.vector_store
                                chat_history = "\n".join([
                                    f"{msg['role'].upper()}: {msg['content']}"
                                    for msg in session.messages
                                ])
                                self.rag_system.setup_qa_chain(
                                    self.vector_store, 
                                    session.session_id,
                                    initial_chat_history=chat_history
                                )
                            else:
                                st.error("❌ Vector store non disponibile.")
                                return

                        result = self.rag_system.query(prompt, session.session_id)

                        st.markdown(result["answer"])

                        if result["sources"]:
                            with st.expander("📚 Fonti"):
                                for source in result["sources"]:
                                    st.markdown(f"**{source['source']}** (Pagina {source['page']})")
                                    st.markdown(f"_{source['content_preview']}_")

                        session.add_message("assistant", result["answer"], result["sources"])
                        self.chat_manager.save_session(session)
                        st.rerun()

                    except Exception as e:
                        error_msg = f"❌ Errore nell'elaborazione: {str(e)}"
                        st.error(error_msg)
                        session.add_message("assistant", error_msg)
                        self.chat_manager.save_session(session)
        
        
    
    def _enhance_prompt_by_mode(self, prompt: str, mode: str) -> str:
        """Arricchisce il prompt in base alla modalità selezionata"""
        if mode == "project_search":
            return f"""
            MODALITÀ RICERCA IDEA PROGETTUALE:
            L'utente sta descrivendo un'idea progettuale e vuole trovare bandi compatibili.
            
            Idea progettuale: {prompt}
            
            Per favore:
            1. Analizza l'idea progettuale descritta
            2. Identifica i settori e le tipologie di intervento rilevanti
            3. Trova i bandi più compatibili con questa idea
            4. Suggerisci requisiti specifici da verificare
            5. Indica scadenze e budget disponibili
            
            Fornisci una risposta strutturata e pratica.
            """
        elif mode == "analysis":
            return f"""
            MODALITÀ ANALISI COMPARATIVA:
            L'utente vuole un'analisi o confronto dettagliato.
            
            Richiesta: {prompt}
            
            Per favore fornisci un'analisi strutturata con:
            1. Confronto dettagliato tra le opzioni
            2. Pro e contro di ciascuna opzione
            3. Raccomandazioni specifiche
            4. Tabelle comparative quando utili
            """
        else:
            return prompt
    
    def _handle_project_search(self, enhanced_prompt: str) -> Dict[str, Any]:
        """Gestisce la ricerca per idea progettuale"""
        try:
            if hasattr(self.rag_system, 'search_by_project_idea') and self.vector_store:
                idea_lines = enhanced_prompt.split('\n')
                project_idea = None
                for line in idea_lines:
                    if line.strip().startswith('Idea progettuale:'):
                        project_idea = line.replace('Idea progettuale:', '').strip()
                        break
                
                if not project_idea:
                    project_idea = enhanced_prompt
                
                results = self.rag_system.search_by_project_idea(project_idea, self.vector_store)
                
                if results:
                    response = f"🔍 **Analisi compatibilità per la tua idea progettuale**\n\n"
                    response += f"**Idea analizzata:** {project_idea}\n\n"
                    response += f"**Trovati {len(results)} bandi potenzialmente compatibili:**\n\n"
                    
                    sources = []
                    for i, result in enumerate(results, 1):
                        response += f"### 📋 Bando {i}: {result['source']}\n"
                        response += f"{result['compatibility_analysis']}\n\n"
                        
                        sources.append({
                            'source': result['source'],
                            'page': result.get('page', 1),
                            'content_preview': result['document_preview'][:200] + "..."
                        })
                    
                    return {
                        'answer': response,
                        'sources': sources
                    }
                else:
                    return {
                        'answer': "⚠️ Non ho trovato bandi specificamente compatibili. Prova a riformulare l'idea.",
                        'sources': []
                    }
            else:
                return self.rag_system.query(enhanced_prompt, st.session_state.chat_session_id)
                
        except Exception as e:
            logger.error(f"Errore nella ricerca idea progettuale: {e}")
            return {
                'answer': f"❌ Errore nella ricerca: {str(e)}",
                'sources': []
            }
    
    def render_summary_table_page(self):
        """Pagina per la tabella di sintesi"""
        st.title("📊 Tabella Riassuntiva")
        
        if not st.session_state.documents:
            st.warning("⚠️ Carica prima alcuni documenti nella sezione 'Caricamento Documenti'")
            return
        
        st.markdown("Tabella di sintesi di tutti i bandi caricati, editabile e scaricabile.")
        
        # Info sulla conversione
        markdown_docs = sum(1 for doc in st.session_state.documents 
                          if doc.metadata.get('content_type') == 'markdown')
        if markdown_docs > 0:
            st.info(f"ℹ️ {markdown_docs} documenti convertiti in Markdown per migliore estrazione dati")
        
        if st.button("📊 Genera Tabella di Sintesi", type="primary"):
            with st.spinner("Generando tabella di sintesi..."):
                try:
                    # Usa metodo migliorato se disponibile e se ci sono documenti Markdown
                    if (hasattr(self.rag_system, 'generate_enhanced_summary_table') and 
                        markdown_docs > 0):
                        summary_data = self.rag_system.generate_enhanced_summary_table(st.session_state.documents)
                    else:
                        summary_data = self.rag_system.generate_summary_table(st.session_state.documents)
                    
                    if summary_data:
                        save_session_state('summary_table', summary_data)
                        st.success("✅ Tabella di sintesi generata!")
                    else:
                        st.warning("⚠️ Nessun dato estratto dai documenti")
                        
                except Exception as e:
                    st.error(f"❌ Errore nella generazione: {str(e)}")
        
        # Mostra la tabella se disponibile
        summary_data = load_session_state('summary_table')
        if summary_data:
            st.markdown("### 📋 Tabella di Sintesi")
            
            df = pd.DataFrame(summary_data)
            
            # Pulisci i dati
            for col in df.columns:
                df[col] = df[col].astype(str)
            
            # Editor della tabella
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "Nome Bando": st.column_config.TextColumn("Nome Bando", width="medium"),
                    "Ente Erogatore": st.column_config.TextColumn("Ente Erogatore", width="medium"),
                    "Scadenza": st.column_config.TextColumn("Scadenza", width="small"),
                    "Budget Totale": st.column_config.TextColumn("Budget Totale", width="medium"),
                    "Importo Max per Progetto": st.column_config.TextColumn("Importo Max", width="medium"),
                    "Settori": st.column_config.TextColumn("Settori", width="large"),
                    "Beneficiari": st.column_config.TextColumn("Beneficiari", width="medium"),
                    "Cofinanziamento %": st.column_config.TextColumn("Cofinanziamento %", width="small"),
                    "Stato": st.column_config.SelectboxColumn("Stato", options=["Aperto", "Chiuso", "Da verificare"], width="small"),
                    "Note": st.column_config.TextColumn("Note", width="large"),
                    "url": st.column_config.LinkColumn("Link al Bando", width="medium"),
                    "source": st.column_config.TextColumn("Nome File", width="medium"),
                    "conversion_method": st.column_config.TextColumn("Metodo Conversione", width="small")
                }
            )
            
            # Pulsanti per l'esportazione
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Esporta in Excel"):
                    try:
                        excel_path = export_to_excel(
                            edited_df.to_dict('records'),
                            "sintesi_bandi.xlsx"
                        )
                        
                        with open(excel_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Scarica Excel",
                                data=f.read(),
                                file_name="sintesi_bandi.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Errore nell'esportazione Excel: {str(e)}")
            
            with col2:
                if st.button("📥 Esporta in CSV"):
                    try:
                        csv_path = export_to_csv(
                            edited_df.to_dict('records'),
                            "sintesi_bandi.csv"
                        )
                        
                        with open(csv_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Scarica CSV",
                                data=f.read(),
                                file_name="sintesi_bandi.csv",
                                mime="text/csv"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Errore nell'esportazione CSV: {str(e)}")
    
    def render_synthesis_document_page(self):
        """Pagina per il documento di sintesi (BONUS)"""
        st.title("📄 Report di Sintesi")
        
        if not st.session_state.documents:
            st.warning("⚠️ Carica prima alcuni documenti nella sezione 'Caricamento Documenti'")
            return
        
        st.markdown("Genera un documento di sintesi completo di tutti i bandi caricati.")
        
        # Raggruppa documenti per file
        docs_by_file = {}
        for doc in st.session_state.documents:
            file_name = doc.metadata.get('source', 'Unknown')
            if file_name not in docs_by_file:
                docs_by_file[file_name] = []
            docs_by_file[file_name].append(doc)
        
        # Opzione di selezione
        st.markdown("### 📋 Selezione Documenti")
        
        synthesis_option = st.radio(
            "Scegli cosa includere nella sintesi:",
            options=["Tutti i file caricati", "File specifico"],
            key="synthesis_option"
        )
        
        selected_documents = []
        
        if synthesis_option == "Tutti i file caricati":
            selected_documents = st.session_state.documents
            st.info(f"ℹ️ Verrà generata la sintesi di tutti i {len(docs_by_file)} file caricati")
        else:
            # Selezione file specifico
            file_names = list(docs_by_file.keys())
            selected_file = st.selectbox(
                "Seleziona il file:",
                options=file_names,
                key="selected_file"
            )
            
            if selected_file:
                selected_documents = docs_by_file[selected_file]
                st.info(f"ℹ️ Verrà generata la sintesi del file: {selected_file}")
        
        if st.button("📄 Genera Documento di Sintesi", type="primary"):
            if not selected_documents:
                st.error("❌ Nessun documento selezionato")
                return
                
            with st.spinner("Generando documento di sintesi..."):
                try:
                    synthesis_prompt = """
                    Crea un documento di sintesi completo (handbook) basato sui seguenti bandi.
                    
                    Il documento deve includere:
                    1. Introduzione generale
                    2. Panoramica dei bandi disponibili
                    3. Analisi dei settori finanziati
                    4. Calendario delle scadenze
                    5. Analisi dei budget disponibili
                    6. Guida ai requisiti comuni
                    7. Consigli per la partecipazione
                    8. Conclusioni
                    
                    Mantieni un tono professionale e informativo.
                    Usa formattazione Markdown per strutturare il documento.
                    """
                    
                    # Prepara il contenuto
                    combined_text = ""
                    
                    for i, doc in enumerate(selected_documents[:15]):
                        combined_text += f"\n\n--- Documento {i+1}: {doc.metadata.get('source', 'Sconosciuto')} ---\n"
                        combined_text += doc.page_content[:2000]
                    
                    full_prompt = f"{synthesis_prompt}\n\nDocumenti:\n{combined_text}"
                    
                    response = self.rag_system.llm.invoke(full_prompt)
                    
                    if hasattr(response, 'content'):
                        content = response.content
                    else:
                        content = str(response)
                    
                    content_str = str(content);
                    
                    st.markdown("### 📄 Documento di Sintesi")
                    st.markdown(content_str)
                    
                    # Genera nome file con timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if synthesis_option == "Tutti i file caricati":
                        filename = f"sintesi_tutti_bandi_{timestamp}.md"
                    else:
                        safe_filename = selected_file.replace('.pdf', '').replace(' ', '_')
                        filename = f"sintesi_{safe_filename}_{timestamp}.md"
                    
                    st.download_button(
                        label="⬇️ Scarica Documento di Sintesi",
                        data=content_str.encode('utf-8'),
                        file_name=filename,
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Errore nella generazione del documento: {str(e)}")
    
    def run(self):
        """Esegue l'applicazione principale"""
        if not self.validate_configuration():
            st.stop()
        
        selected_page = self.render_sidebar()
        
        if selected_page == "📁 Gestione Documenti":
            self.render_file_upload_page()
        elif selected_page == "💬 Chat con AI":
            self.render_chatbot_page()
        elif selected_page == "📊 Tabella Riassuntiva":
            self.render_summary_table_page()
        elif selected_page == "📄 Report di Sintesi":
            self.render_synthesis_document_page()

if __name__ == "__main__":
    app = BandiRAGApp()
    app.run()