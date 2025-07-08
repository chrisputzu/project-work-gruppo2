import streamlit as st
import os
import pandas as pd
from typing import List, Dict, Any
import logging
from pathlib import Path
from datetime import datetime

# Import dei moduli locali
from src.config import Config
from src.document_processor import DocumentProcessor
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
    page_title="Sistema RAG per Bandi Pubblici",
    page_icon="📋",
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
        self.document_processor = DocumentProcessor()
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
        st.sidebar.title("🏛️ Sistema RAG Bandi")
        
        # Informazioni sulla configurazione
        if self.config.use_azure_embeddings():
            st.sidebar.success("✅ Azure OpenAI: Embeddings Dedicato")
        else:
            st.sidebar.success("✅ Azure OpenAI: Embeddings Standard")
        
        # Menu di navigazione
        page = st.sidebar.selectbox(
            "Seleziona una funzione:",
            [
                "📁 Caricamento Documenti",
                "💬 Chatbot Bandi", 
                "🔍 Ricerca per Idea Progettuale",
                "📊 Tabella di Sintesi",
                "📄 Documento di Sintesi (BONUS)"
            ]
        )
        
        # Statistiche
        if st.session_state.processed_files:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📈 Statistiche")
            st.sidebar.metric("Documenti caricati", len(st.session_state.processed_files))
            
            total_size = sum(get_file_stats(f).get('size', 0) for f in st.session_state.processed_files)
            st.sidebar.metric("Dimensione totale", format_file_size(total_size))
            
            # Statistiche e gestione chat
            sessions = self.chat_manager.get_session_list()
            if sessions:
                st.sidebar.metric("Sessioni chat", len(sessions))
                
                # Menu a scorrimento delle chat
                st.sidebar.markdown("### 💬 Le tue chat")
                
                # Pulsante per nuova chat
                if st.sidebar.button("➕ Nuova Chat", key="new_chat_btn"):
                    # Crea una nuova sessione
                    metadata = {
                        "created_by": "user",
                        "documents_count": len(st.session_state.processed_files),
                        "documents": [os.path.basename(f) for f in st.session_state.processed_files]
                    }
                    session_id = self.chat_manager.create_session(metadata)
                    st.session_state.chat_session_id = session_id
                    st.rerun()
                
                # Container scrollabile per le chat
                with st.sidebar.container():
                    for session in sessions:
                        # Crea un expander per ogni chat
                        summary = self.chat_manager.get_session_summary(session['session_id'])
                        if summary:
                            chat_title = f"💭 Chat {datetime.fromisoformat(session['created_at']).strftime('%d/%m/%Y %H:%M')} ({summary['message_count']} msg)"
                        else:
                            chat_title = f"💭 Chat {datetime.fromisoformat(session['created_at']).strftime('%d/%m/%Y %H:%M')}"
                            
                        with st.expander(chat_title):
                            # Preview della chat
                            if summary:
                                st.caption(f"Messaggi: {summary['message_count']}")
                                if summary['keywords']:
                                    st.caption("Ultima domanda:")
                                    st.caption(f"_{summary['keywords'][0]}_")
                            
                            # Bottoni per le azioni
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("💬 Continua", key=f"continue_{session['session_id']}"):
                                    st.session_state.chat_session_id = session['session_id']
                                    self.chat_manager.set_current_session(session['session_id'])
                                    if page != "💬 Chatbot Bandi":
                                        st.session_state.selected_page = "💬 Chatbot Bandi"
                                    st.rerun()
                            
                            with col2:
                                if st.button("📥 Esporta", key=f"export_{session['session_id']}"):
                                    file_path = self.chat_manager.export_session(session['session_id'], "txt")
                                    if file_path:
                                        with open(file_path, "rb") as f:
                                            st.download_button(
                                                label="⬇️ Scarica",
                                                data=f.read(),
                                                file_name=os.path.basename(file_path),
                                                mime="text/plain",
                                                key=f"download_{session['session_id']}"
                                            )
                            
                            with col3:
                                if st.button("🗑️ Elimina", key=f"delete_{session['session_id']}"):
                                    if self.chat_manager.delete_session(session['session_id']):
                                        if st.session_state.chat_session_id == session['session_id']:
                                            st.session_state.chat_session_id = None
                                        st.success("✅ Chat eliminata!")
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
    
    def render_file_upload_page(self):
        """Pagina per il caricamento dei documenti"""
        st.title("📁 Caricamento Documenti Bandi")
        
        st.markdown("""
        Carica i documenti PDF dei bandi pubblici per creare la knowledge base.
        Il sistema supporta il caricamento di più file contemporaneamente.
        """)
        
        # Upload dei file
        uploaded_files = st.file_uploader(
            "Seleziona i file PDF dei bandi",
            type=['pdf'],
            accept_multiple_files=True,
            help="Carica uno o più file PDF contenenti i bandi pubblici"
        )
        
        if uploaded_files:
            # Mostra informazioni sui file caricati
            st.markdown("### 📋 File selezionati:")
            
            for i, file in enumerate(uploaded_files):
                if validate_pdf_file(file):
                    st.success(f"✅ {file.name} - {format_file_size(file.size)}")
                else:
                    st.error(f"❌ {file.name} - File non valido")
            
            # Pulsante per processare
            if st.button("🚀 Processa Documenti", type="primary"):
                self.process_uploaded_files(uploaded_files)
        
        # Mostra file già processati
        if st.session_state.processed_files:
            st.markdown("### 📚 Documenti già processati:")
            
            for file_path in st.session_state.processed_files:
                file_stats = get_file_stats(file_path)
                if file_stats:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.text(f"📄 {file_stats['name']}")
                    with col2:
                        st.text(file_stats['size_formatted'])
                    with col3:
                        st.text(file_stats['created'])
    
    def process_uploaded_files(self, uploaded_files):
        """Processa i file caricati"""
        valid_files = [f for f in uploaded_files if validate_pdf_file(f)]
        
        if not valid_files:
            st.error("❌ Nessun file PDF valido trovato!")
            return
        
        # Barra di progresso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Salva i file
            file_paths = []
            for i, file in enumerate(valid_files):
                status_text.text(f"Salvando {file.name}...")
                file_path = save_uploaded_file(file)
                file_paths.append(file_path)
                progress_bar.progress((i + 1) / (len(valid_files) * 2))
            
            # Processa i documenti
            status_text.text("Processando documenti...")
            self.vector_store = self.document_processor.process_multiple_files(file_paths)
            
            # Configurazione del sistema RAG
            status_text.text("Configurando sistema RAG...")
            
            # Se non c'è una sessione attiva, ne crea una nuova
            if not st.session_state.chat_session_id:
                metadata = {
                    "created_by": "user",
                    "documents_count": len(file_paths),
                    "documents": [os.path.basename(f) for f in file_paths]
                }
                session_id = self.chat_manager.create_session(metadata)
                st.session_state.chat_session_id = session_id
                logger.info(f"Creata nuova sessione per il processamento: {session_id}")
            
            # Configura il sistema RAG con la sessione corrente
            self.rag_system.setup_qa_chain(self.vector_store, st.session_state.chat_session_id)
            
            # Salva lo stato
            st.session_state.processed_files.extend(file_paths)
            st.session_state.vector_store_ready = True
            st.session_state.vector_store = self.vector_store
            
            # Carica tutti i documenti per le funzionalità avanzate
            all_documents = []
            for file_path in file_paths:
                docs = self.document_processor.load_pdf(file_path)
                all_documents.extend(docs)
            
            st.session_state.documents.extend(all_documents)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processamento completato!")
            
            st.success(f"🎉 Processati {len(valid_files)} documenti con successo!")
            
        except Exception as e:
            st.error(f"❌ Errore durante il processamento: {str(e)}")
            logger.error(f"Errore nel processamento: {str(e)}")
            raise
    
    def render_chat_history_page(self):
        """Pagina per la cronologia delle chat"""
        st.title("📝 Cronologia Chat")
        
        # Lista delle sessioni
        sessions = self.chat_manager.get_session_list()
        
        if not sessions:
            st.info("👋 Nessuna chat trovata. Inizia una conversazione nella sezione 'Chatbot Bandi'!")
            return
        
        # Ricerca
        search_query = st.text_input("🔍 Cerca nelle chat", placeholder="Inserisci parole chiave...")
        
        if search_query:
            results = self.chat_manager.search_sessions(search_query)
            if results:
                st.success(f"✅ Trovate {len(results)} chat pertinenti")
                sessions = results
            else:
                st.warning("⚠️ Nessun risultato trovato")
                return
        
        # Mostra le sessioni
        for session_data in sessions:
            session_id = session_data["session_id"]
            summary = self.chat_manager.get_session_summary(session_id)
            
            if summary:
                with st.expander(f"💬 Chat del {datetime.fromisoformat(summary['created_at']).strftime('%d/%m/%Y %H:%M')}"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**Messaggi totali:** {summary['message_count']}")
                        st.markdown(f"**Utente:** {summary['user_messages']} | **Assistente:** {summary['assistant_messages']}")
                    
                    with col2:
                        if st.button("📜 Visualizza", key=f"view_{session_id}"):
                            st.session_state.chat_session_id = session_id
                            self.chat_manager.set_current_session(session_id)
                            st.rerun()
                    
                    with col3:
                        # Menu per l'esportazione
                        export_format = st.selectbox(
                            "Formato",
                            ["json", "txt"],
                            key=f"format_{session_id}"
                        )
                        
                        if st.button("📥 Esporta", key=f"export_{session_id}"):
                            file_path = self.chat_manager.export_session(session_id, export_format)
                            if file_path:
                                with open(file_path, "rb") as f:
                                    st.download_button(
                                        label="⬇️ Scarica",
                                        data=f.read(),
                                        file_name=os.path.basename(file_path),
                                        mime="application/json" if export_format == "json" else "text/plain"
                                    )
                    
                    # Preview dei messaggi
                    if summary['keywords']:
                        st.markdown("**Ultime domande:**")
                        for keyword in summary['keywords']:
                            st.markdown(f"- _{keyword}_")
                    
                    # Opzioni avanzate
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🗑️ Elimina", key=f"delete_{session_id}"):
                            if self.chat_manager.delete_session(session_id):
                                st.success("✅ Chat eliminata!")
                                st.rerun()
                    
                    with col2:
                        if st.button("📋 Copia ID", key=f"copy_{session_id}"):
                            st.code(session_id, language=None)
                            st.success("✅ ID copiato!")
    
    def render_chatbot_page(self):
        """Pagina del chatbot per domande sui bandi"""
        st.title("💬 Chatbot Bandi")
        
        if not st.session_state.vector_store_ready:
            st.warning("⚠️ Carica prima alcuni documenti nella sezione 'Caricamento Documenti'")
            return
        
        # Gestione della sessione di chat
        if not st.session_state.chat_session_id:
            # Crea una nuova sessione
            metadata = {
                "created_by": "user",
                "documents_count": len(st.session_state.processed_files),
                "documents": [os.path.basename(f) for f in st.session_state.processed_files]
            }
            session_id = self.chat_manager.create_session(metadata)
            st.session_state.chat_session_id = session_id
            logger.info(f"Creata nuova sessione chat: {session_id}")
        
        # Ottieni la sessione corrente
        session = self.chat_manager.get_session(st.session_state.chat_session_id)
        if not session:
            st.error("❌ Errore nel caricamento della sessione")
            return
        
        # Debug info
        logger.info(f"Sessione corrente: {session.session_id}, Messaggi: {len(session.messages)}")
        
        # Informazioni sulla chat corrente
        st.caption(f"💭 Chat iniziata il {datetime.fromisoformat(session.created_at).strftime('%d/%m/%Y alle %H:%M')}")
        
        # Ottieni il riepilogo della memoria
        memory_summary = self.rag_system.get_memory_summary(session.session_id)
        if memory_summary["has_memory"]:
            st.caption(f"🧠 Memoria attiva: {memory_summary['message_count']} messaggi")
            
            # Debug della memoria
            logger.info(f"Memoria attiva per sessione {session.session_id}: {memory_summary}")
            if memory_summary["last_interaction"]:
                logger.info(f"Ultima interazione: {memory_summary['last_interaction']}")
        
        # Esempi di domande
        with st.expander("💡 Esempi di domande"):
            st.markdown("""
            - Quali sono le scadenze dei bandi disponibili?
            - Qual è il budget massimo per i progetti?
            - Quali sono i requisiti per partecipare?
            - Chi sono i beneficiari dei bandi?
            - Quali settori sono finanziati?
            """)
        
        # Container per la chat con stile personalizzato
        chat_container = st.container()
        with chat_container:
            # Mostra la cronologia della chat
            for message in session.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Mostra le fonti se disponibili
                    if "sources" in message:
                        with st.expander("📚 Fonti"):
                            for source in message["sources"]:
                                st.markdown(f"**{source['source']}** (Pagina {source['page']})")
                                st.markdown(f"_{source['content_preview']}_")
        
        # Input per nuove domande
        if prompt := st.chat_input("Fai una domanda sui bandi o parliamo..."):
            # Aggiungi la domanda dell'utente
            session.add_message("user", prompt)
            self.chat_manager.save_session(session)
            logger.info(f"Aggiunto messaggio utente alla sessione {session.session_id}: {prompt}")
            
            # Mostra la domanda
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Genera la risposta
            with st.chat_message("assistant"):
                with st.spinner("Elaborando risposta..."):
                    try:
                        # Reinizializza il sistema RAG se necessario
                        if not hasattr(self, 'vector_store') or self.vector_store is None:
                            if st.session_state.vector_store is not None:
                                self.vector_store = st.session_state.vector_store
                                # Passa i messaggi esistenti al setup della catena
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
                                st.error("❌ Vector store non disponibile. Carica prima alcuni documenti.")
                                return
                        
                        # Debug della memoria prima della query
                        memory_before = self.rag_system.get_memory_summary(session.session_id)
                        logger.info(f"Memoria prima della query: {memory_before}")
                        
                        # Esegui la query
                        result = self.rag_system.query(prompt, session.session_id)
                        
                        # Debug della risposta
                        logger.info(f"Risposta ricevuta: {result['answer']}")
                        
                        # Mostra la risposta
                        st.markdown(result["answer"])
                        
                        # Mostra le fonti
                        if result["sources"]:
                            with st.expander("📚 Fonti"):
                                for source in result["sources"]:
                                    st.markdown(f"**{source['source']}** (Pagina {source['page']})")
                                    st.markdown(f"_{source['content_preview']}_")
                        
                        # Salva la risposta nella sessione
                        session.add_message("assistant", result["answer"], result["sources"])
                        self.chat_manager.save_session(session)
                        logger.info(f"Aggiunta risposta assistente alla sessione {session.session_id}")
                        
                        # Debug della memoria dopo la query
                        memory_after = self.rag_system.get_memory_summary(session.session_id)
                        logger.info(f"Memoria dopo la query: {memory_after}")
                        
                        # Forza il refresh della pagina
                        st.rerun()
                        
                    except Exception as e:
                        error_msg = f"❌ Errore nell'elaborazione della domanda: {str(e)}"
                        st.error(error_msg)
                        session.add_message("assistant", error_msg)
                        self.chat_manager.save_session(session)
                        logger.error(f"Errore nella sessione {session.session_id}: {str(e)}")
                        logger.exception("Dettaglio errore:")
        
        # Aggiungi un po' di spazio dopo la chat
        st.markdown("---")
        
        # Statistiche e controlli della chat
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.metric("Messaggi totali", len(session.messages))
            logger.info(f"Totale messaggi nella sessione {session.session_id}: {len(session.messages)}")
        
        with col2:
            user_messages = sum(1 for m in session.messages if m["role"] == "user")
            st.metric("Domande", user_messages)
        
        with col3:
            assistant_messages = sum(1 for m in session.messages if m["role"] == "assistant")
            st.metric("Risposte", assistant_messages)
        
        with col4:
            if st.button("🧹 Pulisci Memoria"):
                # Pulisci la memoria del RAG system
                self.rag_system.clear_memory(session.session_id)
                # Pulisci i messaggi della sessione
                session.messages = []
                self.chat_manager.save_session(session)
                st.success("✅ Chat pulita!")
                logger.info(f"Memoria e messaggi puliti per sessione {session.session_id}")
                st.rerun()
    
    def render_project_search_page(self):
        """Pagina per la ricerca per idea progettuale"""
        st.title("🔍 Ricerca per Idea Progettuale")
        
        if not st.session_state.vector_store_ready:
            st.warning("⚠️ Carica prima alcuni documenti nella sezione 'Caricamento Documenti'")
            return
        
        st.markdown("""
        Inserisci la tua idea progettuale e trova i bandi compatibili.
        """)
        
        # Input per l'idea progettuale
        project_idea = st.text_area(
            "Descrivi la tua idea progettuale:",
            placeholder="Es: Voglio produrre bottiglie d'acqua di plastica riciclata per ridurre l'impatto ambientale...",
            height=100
        )
        
        if st.button("🔍 Cerca Bandi Compatibili", type="primary") and project_idea:
            with st.spinner("Analizzando compatibilità con i bandi..."):
                try:
                    # Reinizializza il sistema se necessario
                    if not hasattr(self, 'vector_store') or self.vector_store is None:
                        if st.session_state.vector_store is not None:
                            self.vector_store = st.session_state.vector_store
                        else:
                            st.error("❌ Vector store non disponibile. Carica prima alcuni documenti.")
                            return
                    
                    if self.vector_store is None:
                        st.error("❌ Vector store non inizializzato. Carica prima alcuni documenti.")
                        return
                    
                    results = self.rag_system.search_by_project_idea(project_idea, self.vector_store)
                    
                    if results:
                        st.success(f"✅ Trovati {len(results)} bandi potenzialmente compatibili")
                        
                        for i, result in enumerate(results, 1):
                            with st.expander(f"📋 Bando {i}: {result['source']}"):
                                st.markdown("**Analisi di Compatibilità:**")
                                st.markdown(result['compatibility_analysis'])
                                
                                st.markdown("**Anteprima Documento:**")
                                st.markdown(f"_{result['document_preview']}_")
                    else:
                        st.warning("⚠️ Nessun bando compatibile trovato per questa idea progettuale")
                        
                except Exception as e:
                    st.error(f"❌ Errore nella ricerca: {str(e)}")
    
    def render_summary_table_page(self):
        """Pagina per la tabella di sintesi"""
        st.title("📊 Tabella di Sintesi Bandi")
        
        if not st.session_state.documents:
            st.warning("⚠️ Carica prima alcuni documenti nella sezione 'Caricamento Documenti'")
            return
        
        st.markdown("""
        Tabella di sintesi di tutti i bandi caricati, editabile e scaricabile.
        """)
        
        # Genera o carica la tabella di sintesi
        if st.button("📊 Genera Tabella di Sintesi", type="primary"):
            with st.spinner("Generando tabella di sintesi..."):
                try:
                    summary_data = self.rag_system.generate_summary_table(st.session_state.documents)
                    
                    if summary_data:
                        # Salva nello stato della sessione
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
            
            # Converti in DataFrame per l'editing
            df = pd.DataFrame(summary_data)
            
            # Pulisci i dati per evitare problemi di tipo
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
                    "Stato": st.column_config.SelectboxColumn("Stato", options=["Aperto", "Chiuso", "N/A"], width="small"),
                    "Note": st.column_config.TextColumn("Note", width="large")
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
        st.title("📄 Documento di Sintesi (BONUS)")
        
        if not st.session_state.documents:
            st.warning("⚠️ Carica prima alcuni documenti nella sezione 'Caricamento Documenti'")
            return
        
        st.markdown("""
        Genera un documento di sintesi (handbook) completo di tutti i bandi caricati.
        """)
        
        if st.button("📄 Genera Documento di Sintesi", type="primary"):
            with st.spinner("Generando documento di sintesi..."):
                try:
                    # Prompt per il documento di sintesi
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
                    """
                    
                    # Combina tutti i documenti
                    combined_text = ""
                    for doc in st.session_state.documents[:10]:  # Limita per evitare token eccessivi
                        combined_text += f"\n\n--- {doc.metadata.get('source', 'Documento')} ---\n"
                        combined_text += doc.page_content[:2000]  # Limita ogni documento
                    
                    full_prompt = f"{synthesis_prompt}\n\nDocumenti:\n{combined_text}"
                    
                    # Genera il documento
                    response = self.rag_system.llm.invoke(full_prompt)
                    
                    # Estrai il contenuto della risposta
                    if hasattr(response, 'content'):
                        content = response.content
                    else:
                        content = str(response)
                    
                    # Assicurati che content sia una stringa
                    if isinstance(content, str):
                        content_str = content
                    else:
                        content_str = str(content)
                    
                    # Mostra il documento
                    st.markdown("### 📄 Documento di Sintesi")
                    st.markdown(content_str)
                    
                    # Pulsante per il download
                    st.download_button(
                        label="⬇️ Scarica Documento di Sintesi",
                        data=content_str.encode('utf-8'),
                        file_name="sintesi_bandi_handbook.md",
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Errore nella generazione del documento: {str(e)}")
    
    def run(self):
        """Esegue l'applicazione principale"""
        # Valida la configurazione
        if not self.validate_configuration():
            st.stop()
        
        # Renderizza la sidebar e ottieni la pagina selezionata
        selected_page = self.render_sidebar()
        
        # Renderizza la pagina selezionata
        if selected_page == "📁 Caricamento Documenti":
            self.render_file_upload_page()
        elif selected_page == "💬 Chatbot Bandi":
            self.render_chatbot_page()
        elif selected_page == "🔍 Ricerca per Idea Progettuale":
            self.render_project_search_page()
        elif selected_page == "📊 Tabella di Sintesi":
            self.render_summary_table_page()
        elif selected_page == "📄 Documento di Sintesi (BONUS)":
            self.render_synthesis_document_page()
        elif selected_page == "📝 Cronologia Chat":
            self.render_chat_history_page()

if __name__ == "__main__":
    app = BandiRAGApp()
    app.run()