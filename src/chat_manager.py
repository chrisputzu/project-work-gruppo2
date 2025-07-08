import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ChatSession:
    """Gestisce una singola sessione di chat"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str, sources: Optional[List[Dict[str, Any]]] = None):
        """Aggiunge un messaggio alla sessione"""
        message: Dict[str, Union[str, List[Dict[str, Any]]]] = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        if sources:
            message["sources"] = sources
        
        self.messages.append(message)
        self.last_updated = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte la sessione in dizionario"""
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Crea una sessione da dizionario"""
        session = cls(data["session_id"])
        session.messages = data.get("messages", [])  # Usa get con default vuoto
        session.created_at = data.get("created_at", datetime.now().isoformat())
        session.last_updated = data.get("last_updated", datetime.now().isoformat())
        session.metadata = data.get("metadata", {})
        return session

class ChatManager:
    """Gestisce tutte le sessioni di chat e la persistenza"""
    
    def __init__(self, storage_dir: str = "chat_history"):
        self.storage_dir = storage_dir
        self.sessions: Dict[str, ChatSession] = {}
        self.current_session_id: Optional[str] = None
        
        # Crea la directory se non esiste
        Path(storage_dir).mkdir(exist_ok=True)
        
        # Carica le sessioni esistenti
        self.load_sessions()
    
    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Crea una nuova sessione di chat"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = ChatSession(session_id)
        
        if metadata:
            session.metadata = metadata
        
        self.sessions[session_id] = session
        self.current_session_id = session_id
        
        # Salva la nuova sessione
        self.save_session(session)
        
        logger.info(f"Creata nuova sessione: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Ottiene una sessione specifica"""
        if session_id not in self.sessions:
            # Prova a caricare dal disco
            file_path = os.path.join(self.storage_dir, f"{session_id}.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        session = ChatSession.from_dict(data)
                        self.sessions[session_id] = session
                        return session
                except Exception as e:
                    logger.error(f"Errore nel caricamento della sessione {session_id}: {str(e)}")
                    return None
            return None
        return self.sessions[session_id]
    
    def get_current_session(self) -> Optional[ChatSession]:
        """Ottiene la sessione corrente"""
        if self.current_session_id:
            return self.get_session(self.current_session_id)
        return None
    
    def set_current_session(self, session_id: str) -> bool:
        """Imposta la sessione corrente"""
        if session_id in self.sessions:
            self.current_session_id = session_id
            return True
        return False
    
    def add_message_to_current_session(
        self, 
        role: str, 
        content: str, 
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Aggiunge un messaggio alla sessione corrente"""
        session = self.get_current_session()
        if session:
            session.add_message(role, content, sources)
            self.save_session(session)
            logger.info(f"Aggiunto messaggio a sessione {session.session_id}")
            return True
        return False
    
    def get_session_list(self) -> List[Dict[str, Any]]:
        """Ottiene la lista delle sessioni con metadati"""
        sessions = []
        for session_id, session in self.sessions.items():
            sessions.append({
                "session_id": session_id,
                "created_at": session.created_at,
                "last_updated": session.last_updated,
                "message_count": len(session.messages),
                "metadata": session.metadata
            })
        
        # Ordina per data di ultimo aggiornamento (più recente prima)
        sessions.sort(key=lambda x: x["last_updated"], reverse=True)
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Elimina una sessione"""
        if session_id in self.sessions:
            # Rimuovi dalla memoria
            del self.sessions[session_id]
            
            # Rimuovi il file
            file_path = os.path.join(self.storage_dir, f"{session_id}.json")
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Eliminata sessione {session_id}")
            except OSError as e:
                logger.error(f"Errore nella rimozione del file {file_path}: {str(e)}")
            
            # Se era la sessione corrente, resetta
            if self.current_session_id == session_id:
                self.current_session_id = None
            
            return True
        return False
    
    def save_session(self, session: ChatSession):
        """Salva una sessione su disco"""
        file_path = os.path.join(self.storage_dir, f"{session.session_id}.json")
        try:
            # Assicurati che la directory esista
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Salva il file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            
            # Aggiorna la cache in memoria
            self.sessions[session.session_id] = session
            
            logger.info(f"Salvata sessione {session.session_id}")
        except Exception as e:
            logger.error(f"Errore nel salvataggio della sessione {session.session_id}: {str(e)}")
            raise
    
    def load_sessions(self):
        """Carica tutte le sessioni da disco"""
        try:
            for file_name in os.listdir(self.storage_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(self.storage_dir, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            session = ChatSession.from_dict(data)
                            self.sessions[session.session_id] = session
                            logger.info(f"Caricata sessione {session.session_id}")
                    except Exception as e:
                        logger.error(f"Errore nel caricamento della sessione {file_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Errore nel caricamento delle sessioni: {str(e)}")
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Ottiene un riepilogo della sessione"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        # Calcola statistiche
        message_count = len(session.messages)
        user_messages = sum(1 for m in session.messages if m["role"] == "user")
        assistant_messages = sum(1 for m in session.messages if m["role"] == "assistant")
        
        # Estrai parole chiave (primi N caratteri di ogni messaggio utente)
        keywords = []
        for msg in session.messages:
            if msg["role"] == "user":
                preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                keywords.append(preview)
        
        return {
            "session_id": session_id,
            "created_at": session.created_at,
            "last_updated": session.last_updated,
            "message_count": message_count,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "keywords": keywords[-3:],  # Ultimi 3 messaggi utente
            "metadata": session.metadata
        }
    
    def export_session(self, session_id: str, format: str = "json") -> Optional[str]:
        """Esporta una sessione in vari formati"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            file_path = os.path.join(self.storage_dir, f"export_{session_id}_{timestamp}.json")
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
                return file_path
            except Exception as e:
                logger.error(f"Errore nell'esportazione JSON della sessione {session_id}: {str(e)}")
        
        elif format == "txt":
            file_path = os.path.join(self.storage_dir, f"export_{session_id}_{timestamp}.txt")
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Chat Session: {session_id}\n")
                    f.write(f"Created: {session.created_at}\n")
                    f.write(f"Last Updated: {session.last_updated}\n\n")
                    
                    for msg in session.messages:
                        f.write(f"[{msg['role'].upper()}] {msg['timestamp']}\n")
                        f.write(f"{msg['content']}\n")
                        if "sources" in msg:
                            f.write("\nSources:\n")
                            for source in msg["sources"]:
                                f.write(f"- {source['source']} (Page {source['page']})\n")
                        f.write("\n" + "-"*50 + "\n\n")
                
                return file_path
            except Exception as e:
                logger.error(f"Errore nell'esportazione TXT della sessione {session_id}: {str(e)}")
        
        return None
    
    def search_sessions(self, query: str) -> List[Dict[str, Any]]:
        """Cerca nelle sessioni per parole chiave"""
        results = []
        
        for session_id, session in self.sessions.items():
            score = 0
            matches = []
            
            # Cerca nei messaggi
            for msg in session.messages:
                if query.lower() in msg["content"].lower():
                    score += 1
                    preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                    matches.append({
                        "role": msg["role"],
                        "preview": preview,
                        "timestamp": msg["timestamp"]
                    })
            
            # Cerca nei metadati
            for key, value in session.metadata.items():
                if query.lower() in str(value).lower():
                    score += 1
            
            if score > 0:
                results.append({
                    "session_id": session_id,
                    "score": score,
                    "matches": matches[:3],  # Primi 3 match
                    "created_at": session.created_at,
                    "last_updated": session.last_updated,
                    "metadata": session.metadata
                })
        
        # Ordina per rilevanza
        results.sort(key=lambda x: x["score"], reverse=True)
        return results 