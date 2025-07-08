import streamlit as st
import os
import pandas as pd
from typing import List, Dict, Any
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    page_title="LombardIA Bandi - Sistema RAG",
    page_icon="🏛️",
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
        # Logo in sidebar
        logo_path = Path("logo/logo_lombardIA.png")
        if logo_path.exists():
            st.sidebar.image(str(logo_path), width=200)
        
        st.sidebar.title("🏛️ LombardIA Bandi")
        
        # Menu di navigazione
        page = st.sidebar.selectbox(
            "Seleziona una funzione:",
            [
                "📁 Caricamento Documenti",
                "💬 LombardIA Bandi (Chat)", 
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
                                    if page != "💬 LombardIA Bandi (Chat)":
                                        st.session_state.selected_page = "💬 LombardIA Bandi (Chat)"
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
    
    def create_chunks_insights(self, file_paths: List[str], total_chunks: int):
        """Crea insights grafici sui chunks processati"""
        if not file_paths:
            return
        
        # Calcola statistiche per file
        file_stats = []
        total_pages = 0
        
        for file_path in file_paths:
            try:
                # Carica il documento per ottenere statistiche
                documents = self.document_processor.load_pdf(file_path)
                chunks = self.document_processor.process_documents(documents)
                
                file_info = {
                    'nome': os.path.basename(file_path),
                    'pagine': len(documents),
                    'chunks': len(chunks),
                    'size_kb': os.path.getsize(file_path) / 1024
                }
                file_stats.append(file_info)
                total_pages += len(documents)
                
            except Exception as e:
                logger.error(f"Errore nel calcolo statistiche per {file_path}: {e}")
                continue
        
        if not file_stats:
            return
        
        # Crea i grafici
        st.markdown("## 📊 Insights sui Documenti Processati")
        
        # Metriche principali
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Documenti", len(file_paths))
        
        with col2:
            st.metric("📃 Pagine Totali", total_pages)
        
        with col3:
            st.metric("🧩 Chunks Totali", total_chunks)
        
        with col4:
            avg_chunks_per_page = total_chunks / total_pages if total_pages > 0 else 0
            st.metric("📊 Chunks/Pagina", f"{avg_chunks_per_page:.1f}")
        
        # Grafici
        col1, col2 = st.columns(2)
        
        with col1:
            # Grafico a barre: Chunks per documento
            df_chunks = pd.DataFrame(file_stats)
            fig_bar = px.bar(
                df_chunks, 
                x='nome', 
                y='chunks', 
                title='🧩 Chunks per Documento',
                labels={'chunks': 'Numero di Chunks', 'nome': 'Documento'}
            )
            fig_bar.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Grafico a torta: Distribuzione chunks
            fig_pie = px.pie(
                df_chunks, 
                values='chunks', 
                names='nome',
                title='🥧 Distribuzione Chunks per Documento'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Grafico correlazione dimensione-chunks
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot: Dimensione vs Chunks
            fig_scatter = px.scatter(
                df_chunks, 
                x='size_kb', 
                y='chunks',
                size='pagine',
                hover_name='nome',
                title='📈 Dimensione File vs Numero di Chunks',
                labels={'size_kb': 'Dimensione (KB)', 'chunks': 'Numero di Chunks'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Istogramma: Distribuzione dimensioni
            fig_hist = px.histogram(
                df_chunks, 
                x='size_kb', 
                nbins=10,
                title='📊 Distribuzione Dimensioni File',
                labels={'size_kb': 'Dimensione (KB)', 'count': 'Numero di File'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Tabella dettagliata
        st.markdown("### 📋 Dettaglio per Documento")
        
        # Prepara i dati per la tabella
        display_df = df_chunks.copy()
        display_df['size_mb'] = display_df['size_kb'] / 1024
        display_df['chunks_per_page'] = display_df['chunks'] / display_df['pagine']
        
        # Rinomina colonne per display
        display_df = display_df.rename(columns={
            'nome': 'Documento',
            'pagine': 'Pagine',
            'chunks': 'Chunks',
            'size_mb': 'Dimensione (MB)',
            'chunks_per_page': 'Chunks/Pagina'
        })
        
        # Formatta i numeri
        display_df['Dimensione (MB)'] = display_df['Dimensione (MB)'].round(2)
        display_df['Chunks/Pagina'] = display_df['Chunks/Pagina'].round(1)
        
        # Mostra tabella
        st.dataframe(
            display_df[['Documento', 'Pagine', 'Chunks', 'Dimensione (MB)', 'Chunks/Pagina']],
            use_container_width=True
        )
        
        # Insights automatici
        st.markdown("### 🔍 Insights Automatici")
        
        # Documento con più chunks
        max_chunks_doc = df_chunks.loc[df_chunks['chunks'].idxmax()]
        st.info(f"📄 **Documento più complesso**: {max_chunks_doc['nome']} con {max_chunks_doc['chunks']} chunks")
        
        # Documento più grande
        max_size_doc = df_chunks.loc[df_chunks['size_kb'].idxmax()]
        st.info(f"📊 **Documento più grande**: {max_size_doc['nome']} ({max_size_doc['size_kb']/1024:.1f} MB)")
        
        # Efficienza media
        avg_efficiency = df_chunks['chunks'].sum() / df_chunks['pagine'].sum()
        st.info(f"⚡ **Efficienza chunking**: {avg_efficiency:.1f} chunks per pagina in media")
    
    def process_data_folder(self):
        """Processa tutti i file PDF dalla cartella data"""
        data_path = Path(self.config.DATA_DIR)
        
        if not data_path.exists():
            st.error(f"❌ La cartella {self.config.DATA_DIR} non esiste!")
            return []
        
        # Trova tutti i file PDF nella cartella data
        pdf_files = list(data_path.glob("*.pdf"))
        
        if not pdf_files:
            st.warning(f"⚠️ Nessun file PDF trovato nella cartella {self.config.DATA_DIR}")
            return []
        
        # Filtra i file già processati
        # Normalizza i percorsi per confrontarli correttamente
        already_processed_normalized = set()
        for processed_file in st.session_state.processed_files:
            # Converti in Path e ottieni il percorso assoluto normalizzato
            try:
                normalized_path = Path(processed_file).resolve()
                already_processed_normalized.add(str(normalized_path))
            except:
                # Se il percorso non è valido, usa il nome del file
                already_processed_normalized.add(os.path.basename(processed_file))
        
        new_files = []
        for pdf_file in pdf_files:
            # Normalizza anche il percorso del file da controllare
            normalized_pdf = str(pdf_file.resolve())
            if normalized_pdf not in already_processed_normalized:
                # Controlla anche per nome file nel caso di file caricati
                if os.path.basename(str(pdf_file)) not in already_processed_normalized:
                    new_files.append(pdf_file)
        
        if not new_files:
            st.info("ℹ️ Tutti i file nella cartella data sono già stati processati")
            return []
        
        return new_files
    
    def render_file_upload_page(self):
        """Pagina per il caricamento dei documenti"""
        # Header con logo
        col1, col2 = st.columns([1, 4])
        with col1:
            logo_path = Path("logo/logo_lombardIA.png")
            if logo_path.exists():
                st.image(str(logo_path), width=120)
        with col2:
            st.title("📁 Caricamento Documenti Bandi")
        
        st.markdown("""
        Carica i documenti PDF dei bandi pubblici per creare la knowledge base.
        Il sistema supporta il caricamento di più file contemporaneamente.
        """)
        
        # Sezione per processare la cartella data
        st.markdown("### 🗂️ Processa Cartella Data")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Controlla quanti file sono disponibili nella cartella data
            available_files = self.process_data_folder()
            if available_files:
                st.info(f"📁 Trovati {len(available_files)} nuovi file PDF nella cartella '{self.config.DATA_DIR}'")
                
                # Mostra l'anteprima dei file disponibili
                with st.expander("👀 Anteprima file disponibili"):
                    for file in available_files[:10]:  # Mostra max 10 file
                        st.text(f"📄 {file.name}")
                    if len(available_files) > 10:
                        st.text(f"... e altri {len(available_files) - 10} file")
            else:
                st.info(f"📁 Nessun nuovo file PDF da processare nella cartella '{self.config.DATA_DIR}'")
        
        with col2:
            if st.button("🚀 Processa Tutti", type="primary", disabled=not available_files):
                self.process_files_from_data_folder(available_files)
        
        st.markdown("---")
        
        # Sezione per upload manuale
        st.markdown("### 📤 Caricamento Manuale")
        st.markdown("Oppure carica documenti aggiuntivi manualmente:")
        
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
            
            # Separa i file per origine
            data_folder_files = []
            uploaded_files = []
            
            for file_path in st.session_state.processed_files:
                # Controlla se il file proviene dalla cartella data
                if file_path.startswith(self.config.DATA_DIR):
                    data_folder_files.append(file_path)
                else:
                    uploaded_files.append(file_path)
            
            # Mostra file dalla cartella data
            if data_folder_files:
                st.markdown("#### 🗂️ Dalla cartella data:")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.text(f"📁 {len(data_folder_files)} file processati dalla cartella data")
                
                with col2:
                    if st.button("🔄 Reset Data", key="reset_data_files"):
                        # Rimuovi solo i file dalla cartella data
                        st.session_state.processed_files = [
                            f for f in st.session_state.processed_files 
                            if not f.startswith(self.config.DATA_DIR)
                        ]
                        st.success("✅ File dalla cartella data rimossi. Ricarica la pagina per riprocessarli.")
                        st.rerun()
                
                # Mostra alcuni file di esempio
                for file_path in data_folder_files[:5]:  # Mostra max 5 file
                    file_stats = get_file_stats(file_path)
                    if file_stats:
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.text(f"📄 {file_stats['name']}")
                        with col2:
                            st.text(file_stats['size_formatted'])
                        with col3:
                            st.text(file_stats['created'])
                
                if len(data_folder_files) > 5:
                    st.text(f"... e altri {len(data_folder_files) - 5} file")
            
            # Mostra file caricati manualmente
            if uploaded_files:
                st.markdown("#### 📤 Caricati manualmente:")
                for file_path in uploaded_files:
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
        """Processa i file caricati manualmente"""
        valid_files = [f for f in uploaded_files if validate_pdf_file(f)]
        
        if not valid_files:
            st.error("❌ Nessun file PDF valido trovato!")
            return
        
        # Barra di progresso
        progress_bar = st.progress(0)
        status_text = st.empty()
        progress_info = st.empty()
        
        def update_progress(message):
            """Callback per aggiornare il progresso"""
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
            
            # Informazioni sui file da processare
            progress_info.info(f"📊 Processamento batch: {len(file_paths)} file caricati, batch size: {self.config.BATCH_SIZE}")
            
            # Processa i documenti usando il metodo batch
            status_text.text("Processando documenti in batch...")
            progress_bar.progress(0.25)
            
            self.vector_store = self.document_processor.process_multiple_files_batch(
                file_paths,
                progress_callback=update_progress
            )
            
            progress_bar.progress(0.75)
            
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
            
            progress_info.empty()
            st.success(f"🎉 Processati {len(valid_files)} documenti con successo!")
            
            # Mostra insights sui chunks processati
            if hasattr(self, 'vector_store') and self.vector_store:
                # Calcola il numero totale di chunks dal vector store
                total_chunks = self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else len(all_documents)
                self.create_chunks_insights(file_paths, total_chunks)
            
        except Exception as e:
            progress_info.empty()
            st.error(f"❌ Errore durante il processamento: {str(e)}")
            logger.error(f"Errore nel processamento: {str(e)}")
            
            if "429" in str(e) or "rate limit" in str(e).lower():
                st.warning("💡 **Rate limit raggiunto**: Il sistema batch includerà retry automatici. Attendi...")
            raise
    
    def process_files_from_data_folder(self, files_to_process):
        """Processa i file dalla cartella data"""
        if not files_to_process:
            st.warning("⚠️ Nessun file da processare nella cartella data.")
            return
        
        # Barra di progresso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Contatori per il progresso dettagliato
        progress_info = st.empty()
        
        def update_progress(message):
            """Callback per aggiornare il progresso"""
            status_text.text(message)
            logger.info(f"Progress: {message}")
        
        try:
            # Usa direttamente i percorsi dei file dalla cartella data
            file_paths = []
            for i, file_path in enumerate(files_to_process):
                status_text.text(f"Preparando {file_path.name}...")
                # Converti Path in stringa
                file_paths.append(str(file_path))
                progress_bar.progress((i + 1) / (len(files_to_process) * 4))  # 25% per preparazione
            
            # Informazioni sui file da processare
            progress_info.info(f"📊 Processamento in batch: {len(file_paths)} file, batch size: {self.config.BATCH_SIZE}")
            
            # Processa i documenti usando il nuovo metodo batch
            status_text.text("Processando documenti in batch per evitare rate limits...")
            progress_bar.progress(0.25)  # 25% completato
            
            self.vector_store = self.document_processor.process_multiple_files_batch(
                file_paths, 
                progress_callback=update_progress
            )
            
            progress_bar.progress(0.75)  # 75% completato
            
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
            status_text.text("Caricando documenti per funzionalità avanzate...")
            all_documents = []
            for file_path in file_paths:
                docs = self.document_processor.load_pdf(file_path)
                all_documents.extend(docs)
            
            st.session_state.documents.extend(all_documents)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processamento completato!")
            
            # Rimuovi le informazioni di progresso e mostra il risultato finale
            progress_info.empty()
            st.success(f"🎉 Processati {len(files_to_process)} documenti dalla cartella data con successo!")
            st.info(f"📈 Vector store creato con migliaia di chunks utilizzando processamento batch intelligente")
            
            # Mostra insights sui chunks processati
            if hasattr(self, 'vector_store') and self.vector_store:
                # Calcola il numero totale di chunks dal vector store
                total_chunks = self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else len(all_documents)
                self.create_chunks_insights(file_paths, total_chunks)
            
            st.rerun()
            
        except Exception as e:
            progress_info.empty()
            st.error(f"❌ Errore durante il processamento: {str(e)}")
            logger.error(f"Errore nel processamento: {str(e)}")
            
            # Mostra suggerimenti in caso di errore
            if "429" in str(e) or "rate limit" in str(e).lower():
                st.warning("💡 **Suggerimento**: Se l'errore persiste, prova a:")
                st.markdown("""
                - Attendere qualche minuto prima di riprovare
                - Contattare il supporto Azure per aumentare i limiti
                - Processare meno file alla volta
                """)
            raise
    
    def render_chat_history_page(self):
        """Pagina per la cronologia delle chat"""
        st.title("📝 Cronologia Chat")
        
        # Lista delle sessioni
        sessions = self.chat_manager.get_session_list()
        
        if not sessions:
            st.info("👋 Nessuna chat trovata. Inizia una conversazione nella sezione 'LombardIA Bandi (Chat)'!")
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
        # Header con logo
        col1, col2 = st.columns([1, 4])
        with col1:
            logo_path = Path("logo/logo_lombardIA.png")
            if logo_path.exists():
                st.image(str(logo_path), width=120)
        with col2:
            st.title("💬 LombardIA Bandi")
        
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
        
        # Modalità di conversazione
        st.markdown("### 🎯 Modalità di Conversazione")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💬 Domanda Generale", key="mode_general", help="Fai domande generali sui bandi"):
                st.session_state.chat_mode = "general"
                st.rerun()
        
        with col2:
            if st.button("🔍 Ricerca Idea Progettuale", key="mode_project", help="Trova bandi compatibili con la tua idea"):
                st.session_state.chat_mode = "project_search"
                st.rerun()
        
        with col3:
            if st.button("📊 Analisi Comparativa", key="mode_analysis", help="Confronta bandi e analisi"):
                st.session_state.chat_mode = "analysis"
                st.rerun()
        
        # Inizializza la modalità se non esiste
        if 'chat_mode' not in st.session_state:
            st.session_state.chat_mode = "general"
        
        # Mostra la modalità attiva
        mode_labels = {
            "general": "💬 Modalità Generale",
            "project_search": "🔍 Ricerca Idea Progettuale", 
            "analysis": "📊 Analisi Comparativa"
        }
        
        st.info(f"**Modalità attiva**: {mode_labels.get(st.session_state.chat_mode, 'Generale')}")
        
        # Esempi di domande basati sulla modalità
        with st.expander("💡 Esempi di domande"):
            if st.session_state.chat_mode == "general":
                st.markdown("""
                **Domande Generali:**
                - Quali sono le scadenze dei bandi disponibili?
                - Qual è il budget massimo per i progetti?
                - Quali sono i requisiti per partecipare?
                - Chi sono i beneficiari dei bandi?
                - Quali settori sono finanziati?
                """)
            elif st.session_state.chat_mode == "project_search":
                st.markdown("""
                **Ricerca per Idea Progettuale:**
                - "Voglio sviluppare un'app per il turismo sostenibile"
                - "Idea: produzione di packaging ecosostenibile per alimenti"
                - "Progetto di digitalizzazione per piccole imprese artigiane"
                - "Startup nel settore energie rinnovabili per comunità locali"
                - "Piattaforma e-commerce per prodotti locali"
                """)
            else:  # analysis
                st.markdown("""
                **Analisi Comparativa:**
                - Confronta i bandi per startup tecnologiche
                - Analizza le differenze tra bandi regionali e nazionali
                - Quale bando ha il maggior budget per l'innovazione?
                - Confronto requisiti di partecipazione tra diversi bandi
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
        
        # Input personalizzato in base alla modalità
        placeholder_text = {
            "general": "Fai una domanda sui bandi o parliamo...",
            "project_search": "Descrivi la tua idea progettuale per trovare bandi compatibili...",
            "analysis": "Chiedi un'analisi o confronto tra bandi..."
        }
        
        # Input per nuove domande
        if prompt := st.chat_input(placeholder_text.get(st.session_state.chat_mode, "Scrivi qui...")):
            # Aggiungi la domanda dell'utente
            session.add_message("user", prompt)
            self.chat_manager.save_session(session)
            logger.info(f"Aggiunto messaggio utente alla sessione {session.session_id}: {prompt}")
            
            # Mostra la domanda
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Genera la risposta in base alla modalità
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
                        
                        # Personalizza la query in base alla modalità
                        enhanced_prompt = self._enhance_prompt_by_mode(prompt, st.session_state.chat_mode)
                        
                        # Esegui la query
                        if st.session_state.chat_mode == "project_search":
                            # Usa il metodo di ricerca per idea progettuale
                            result = self._handle_project_search(enhanced_prompt)
                        else:
                            # Usa il metodo normale
                            result = self.rag_system.query(enhanced_prompt, session.session_id)
                        
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
        else:  # general
            return prompt
    
    def _handle_project_search(self, enhanced_prompt: str) -> Dict[str, Any]:
        """Gestisce la ricerca per idea progettuale"""
        try:
            # Usa il metodo del RAG system
            if hasattr(self.rag_system, 'search_by_project_idea') and self.vector_store:
                # Estrai l'idea progettuale dal prompt
                idea_lines = enhanced_prompt.split('\n')
                project_idea = None
                for line in idea_lines:
                    if line.strip().startswith('Idea progettuale:'):
                        project_idea = line.replace('Idea progettuale:', '').strip()
                        break
                
                if not project_idea:
                    # Se non trovata, usa tutto il prompt
                    project_idea = enhanced_prompt
                
                # Cerca bandi compatibili
                results = self.rag_system.search_by_project_idea(project_idea, self.vector_store)
                
                if results:
                    # Formatta la risposta
                    response = f"🔍 **Analisi compatibilità per la tua idea progettuale**\n\n"
                    response += f"**Idea analizzata:** {project_idea}\n\n"
                    response += f"**Trovati {len(results)} bandi potenzialmente compatibili:**\n\n"
                    
                    sources = []
                    for i, result in enumerate(results, 1):
                        response += f"### 📋 Bando {i}: {result['source']}\n"
                        response += f"{result['compatibility_analysis']}\n\n"
                        
                        # Aggiungi alle fonti
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
                        'answer': "⚠️ Non ho trovato bandi specificamente compatibili con questa idea progettuale. Prova a riformulare l'idea o fai domande più generali sui settori di interesse.",
                        'sources': []
                    }
            else:
                # Fallback al metodo normale
                return self.rag_system.query(enhanced_prompt, st.session_state.chat_session_id)
                
        except Exception as e:
            logger.error(f"Errore nella ricerca idea progettuale: {e}")
            return {
                'answer': f"❌ Errore nella ricerca per idea progettuale: {str(e)}",
                'sources': []
            }
    
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
                    "Note": st.column_config.TextColumn("Note", width="large"),
                    "url": st.column_config.LinkColumn("Link al Bando", width="medium"),
                    "source": st.column_config.TextColumn("Nome File", width="medium")
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
        elif selected_page == "💬 LombardIA Bandi (Chat)":
            self.render_chatbot_page()
        elif selected_page == "📊 Tabella di Sintesi":
            self.render_summary_table_page()
        elif selected_page == "📄 Documento di Sintesi (BONUS)":
            self.render_synthesis_document_page()
        elif selected_page == "📝 Cronologia Chat":
            self.render_chat_history_page()

if __name__ == "__main__":
    app = BandiRAGApp()
    app.run()