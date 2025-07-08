import os
import logging
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import streamlit as st
from datetime import datetime

logger = logging.getLogger(__name__)

def setup_logging():
    """Configura il logging per l'applicazione"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join("logs", "app.log")),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Logging configurato nella directory logs/")

def create_directories():
    """Crea le directory necessarie per l'applicazione"""
    directories = [
        "data",
        "vector_store", 
        "uploads",
        "exports",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Directory {directory} creata/verificata")

def save_uploaded_file(uploaded_file, upload_dir: str = "uploads") -> str:
    """Salva un file caricato e restituisce il percorso"""
    try:
        Path(upload_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        file_path = os.path.join(upload_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"File salvato: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Errore nel salvataggio del file: {str(e)}")
        raise

def export_to_excel(data: List[Dict[str, Any]], filename: str = "export.xlsx") -> str:
    """Esporta i dati in un file Excel"""
    try:
        Path("exports").mkdir(exist_ok=True)
        filepath = os.path.join("exports", filename)
        
        df = pd.DataFrame(data)
        df.to_excel(filepath, index=False)
        
        logger.info(f"Dati esportati in: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Errore nell'esportazione Excel: {str(e)}")
        raise

def export_to_csv(data: List[Dict[str, Any]], filename: str = "export.csv") -> str:
    """Esporta i dati in un file CSV"""
    try:
        Path("exports").mkdir(exist_ok=True)
        filepath = os.path.join("exports", filename)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Dati esportati in: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Errore nell'esportazione CSV: {str(e)}")
        raise

def format_file_size(size_bytes: float) -> str:
    """Formatta la dimensione del file in modo leggibile"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def validate_pdf_file(file) -> bool:
    """Valida se il file è un PDF"""
    if file is None:
        return False
    
    if not file.name.lower().endswith('.pdf'):
        return False
    
    file_bytes = file.read(4)
    file.seek(0)  # Reset del puntatore
    
    return file_bytes.startswith(b'%PDF')

def save_session_state(key: str, value: Any):
    """Salva un valore nello stato della sessione Streamlit"""
    if 'session_data' not in st.session_state:
        st.session_state.session_data = {}
    
    st.session_state.session_data[key] = value

def load_session_state(key: str, default: Any = None) -> Any:
    """Carica un valore dallo stato della sessione Streamlit"""
    if 'session_data' not in st.session_state:
        st.session_state.session_data = {}
    
    return st.session_state.session_data.get(key, default)

def clear_session_state():
    """Pulisce lo stato della sessione"""
    if 'session_data' in st.session_state:
        st.session_state.session_data.clear()

def get_file_stats(file_path: str) -> Dict[str, Any]:
    """Ottiene statistiche su un file"""
    try:
        stat = os.stat(file_path)
        return {
            'size': stat.st_size,
            'size_formatted': format_file_size(stat.st_size),
            'created': datetime.fromtimestamp(stat.st_ctime).strftime("%d/%m/%Y %H:%M"),
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime("%d/%m/%Y %H:%M"),
            'name': os.path.basename(file_path),
            'extension': os.path.splitext(file_path)[1].lower()
        }
    except:
        return {}