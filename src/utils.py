import os
import logging
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import streamlit as st
from datetime import datetime
import json

logger = logging.getLogger(__name__)

def setup_logging():
    """Configura il logging per l'applicazione"""
    # Crea la directory logs se non esiste
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configura il logging
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
        # Crea la directory se non esiste
        Path(upload_dir).mkdir(exist_ok=True)
        
        # Genera un nome file unico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        file_path = os.path.join(upload_dir, filename)
        
        # Salva il file
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
        # Crea la directory exports se non esiste
        Path("exports").mkdir(exist_ok=True)
        
        # Genera il percorso completo
        filepath = os.path.join("exports", filename)
        
        # Converti in DataFrame e salva
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
        # Crea la directory exports se non esiste
        Path("exports").mkdir(exist_ok=True)
        
        # Genera il percorso completo
        filepath = os.path.join("exports", filename)
        
        # Converti in DataFrame e salva
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
    
    # Controlla l'estensione
    if not file.name.lower().endswith('.pdf'):
        return False
    
    # Controlla i magic bytes del PDF
    file_bytes = file.read(4)
    file.seek(0)  # Reset del puntatore
    
    return file_bytes.startswith(b'%PDF')

def clean_text(text: str) -> str:
    """Pulisce il testo rimuovendo caratteri indesiderati"""
    if not text:
        return ""
    
    # Rimuovi caratteri di controllo
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Rimuovi spazi multipli
    text = ' '.join(text.split())
    
    return text

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Estrae parole chiave dal testo"""
    if not text:
        return []
    
    # Parole comuni da escludere
    stop_words = {
        'il', 'la', 'di', 'che', 'e', 'a', 'da', 'in', 'un', 'è', 'per', 'una', 'sono', 'con', 'non', 'si', 'le', 'dei', 'del', 'al', 'alla', 'delle', 'dell', 'gli', 'lo', 'su', 'nel', 'nella', 'o', 'se', 'ma', 'anche', 'come', 'più', 'può', 'essere', 'stato', 'tutti', 'dalla', 'questo', 'questa', 'hanno', 'aveva', 'quanto', 'tra', 'ogni', 'quando', 'dove', 'dopo', 'prima', 'stesso', 'stesso', 'altri', 'altre', 'tutto', 'tutto', 'molto', 'bene', 'senza', 'fare', 'anni', 'anno', 'tempo', 'volta', 'sempre', 'proprio', 'sotto', 'sopra', 'ancora', 'così', 'qui', 'quindi', 'poi', 'oggi', 'ieri', 'domani'
    }
    
    # Estrai parole (solo lettere, minimo 3 caratteri)
    words = []
    for word in text.lower().split():
        word = ''.join(char for char in word if char.isalpha())
        if len(word) >= 3 and word not in stop_words:
            words.append(word)
    
    # Conta le occorrenze
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    # Ordina per frequenza e restituisci le top N
    keywords = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in keywords[:max_keywords]]

def create_download_link(file_path: str, link_text: str = "Scarica file") -> str:
    """Crea un link per il download di un file"""
    if not os.path.exists(file_path):
        return ""
    
    with open(file_path, "rb") as f:
        data = f.read()
    
    # Determina il tipo MIME
    if file_path.endswith('.xlsx'):
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif file_path.endswith('.csv'):
        mime_type = 'text/csv'
    elif file_path.endswith('.pdf'):
        mime_type = 'application/pdf'
    else:
        mime_type = 'application/octet-stream'
    
    st.download_button(
        label=link_text,
        data=data,
        file_name=os.path.basename(file_path),
        mime=mime_type
    )
    return "Download link created"

def format_currency(amount: str) -> str:
    """Formatta un importo in valuta"""
    try:
        # Rimuovi caratteri non numerici eccetto virgole e punti
        clean_amount = ''.join(char for char in amount if char.isdigit() or char in '.,')
        
        # Converti in numero
        if ',' in clean_amount and '.' in clean_amount:
            # Formato europeo: 1.234.567,89
            clean_amount = clean_amount.replace('.', '').replace(',', '.')
        elif ',' in clean_amount:
            # Potrebbe essere formato europeo: 1234,89
            clean_amount = clean_amount.replace(',', '.')
        
        num_amount = float(clean_amount)
        
        # Formatta in euro
        return f"€ {num_amount:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        
    except:
        return amount

def parse_date(date_str: str) -> str:
    """Parsing intelligente delle date"""
    if not date_str or date_str.strip().lower() in ['n/a', 'non disponibile', 'nd']:
        return "N/A"
    
    # Formati comuni delle date
    date_formats = [
        "%d/%m/%Y",
        "%d-%m-%Y", 
        "%Y-%m-%d",
        "%d/%m/%y",
        "%d-%m-%y",
        "%d.%m.%Y",
        "%d.%m.%y"
    ]
    
    date_str = date_str.strip()
    
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            return parsed_date.strftime("%d/%m/%Y")
        except:
            continue
    
    return date_str

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
