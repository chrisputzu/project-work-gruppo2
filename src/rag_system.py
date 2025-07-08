import os
import json
from typing import List, Dict, Any, Union
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from pydantic import SecretStr
from src.config import Config
from src.document_processor import DocumentProcessor
from src.chat_manager import ChatManager
import logging

logger = logging.getLogger(__name__)

class RAGSystem:
    """Sistema RAG per interrogare i bandi"""
    
    def __init__(self):
        self.config = Config()
        self.document_processor = DocumentProcessor()
        self.llm = self._get_llm()
        self.qa_chain = None
        self.chat_histories: Dict[str, ChatMessageHistory] = {}
        self.memories: Dict[str, ConversationBufferMemory] = {} # Aggiunto per gestire la memoria per sessione
    
    def _get_llm(self):
        """Ottiene il modello LLM appropriato"""
        logger.info("Inizializzando Azure OpenAI Chat")
        return AzureChatOpenAI(
            azure_endpoint=self.config.AZURE_ENDPOINT,
            api_key=SecretStr(self.config.AZURE_API_KEY) if self.config.AZURE_API_KEY else None,
            api_version=self.config.AZURE_API_VERSION,
            azure_deployment=self.config.DEPLOYMENT_NAME,
            temperature=0.7,
            max_tokens=self.config.MAX_TOKENS
        )
    
    def setup_qa_chain(self, vector_store, session_id: str, initial_chat_history: str = ""):
        """Configura la catena QA con memoria"""
        
        # Template per le domande sui bandi
        qa_template = """
        Sei un assistente amichevole ed esperto nell'analisi di bandi pubblici e finanziamenti.
        Il tuo compito è aiutare l'utente a trovare e comprendere i bandi più adatti alle sue esigenze.
        
        CRONOLOGIA DELLA CONVERSAZIONE:
        {chat_history}
        
        CONTESTO SUI BANDI:
        {context}
        
        DOMANDA ATTUALE: {question}
        
        ISTRUZIONI OBBLIGATORIE:
        1. Usa SEMPRE le informazioni dalla cronologia della conversazione per personalizzare le risposte
        2. Se l'utente ha fornito il suo nome in precedenza, usalo sempre
        3. Fai riferimento a domande e risposte precedenti quando rilevante
        4. Mantieni un tono amichevole e professionale
        5. Usa le informazioni dai bandi solo se pertinenti alla domanda
        
        RISPOSTA:
        """
        
        # Crea una nuova memoria per la sessione
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Se c'è una chat history iniziale, la aggiungiamo alla memoria
        if initial_chat_history:
            memory.chat_memory.add_ai_message(initial_chat_history)
        
        # Crea la catena QA
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            memory=memory,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template=qa_template,
                    input_variables=["context", "chat_history", "question"]
                )
            },
            return_source_documents=True,
            verbose=True
        )
        
        # Salva la memoria
        self.memories[session_id] = memory
        
        logger.info(f"Catena QA configurata con successo per sessione {session_id}")
        if initial_chat_history:
            logger.info(f"Caricata chat history iniziale: {initial_chat_history[:100]}...")
    
    def _convert_chat_manager_messages(self, messages: List[Dict[str, Any]]) -> List[Union[HumanMessage, AIMessage]]:
        """Converte i messaggi dal formato del chat manager al formato di LangChain"""
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        return langchain_messages
    
    def _analyze_user_intent(self, question: str) -> bool:
        """Analizza l'intento dell'utente per capire se sta chiedendo informazioni sui bandi"""
        # Prompt per l'analisi dell'intento
        intent_prompt = """
        Analizza il seguente messaggio e determina se l'utente sta chiedendo informazioni specifiche sui bandi/documenti o sta solo chattando.
        
        Messaggio utente: {question}
        
        Esempi di domande sui bandi:
        - Quali sono i requisiti per partecipare?
        - Qual è la scadenza del bando?
        - Quanto è il budget disponibile?
        - Chi sono i beneficiari?
        - Come si presenta la domanda?
        
        Esempi di chat generale:
        - Ciao, come stai?
        - Mi chiamo Mario
        - Grazie dell'aiuto
        - Non ho capito
        - Puoi ripetere?
        
        Rispondi SOLO con "True" se la domanda è sui bandi, "False" se è chat generale.
        """
        
        try:
            # Chiedi al modello di analizzare l'intento
            response = self.llm.invoke(
                intent_prompt.format(question=question)
            )
            
            # Estrai la risposta
            if hasattr(response, 'content'):
                result = response.content.strip().lower()
            else:
                result = str(response).strip().lower()
            
            return result == "true"
            
        except Exception as e:
            logger.error(f"Errore nell'analisi dell'intento: {str(e)}")
            # In caso di errore, assumiamo che sia una domanda sui bandi
            return True
    
    def query(self, question: str, session_id: str) -> Dict[str, Any]:
        """Esegue una query sul sistema RAG"""
        if not self.qa_chain:
            raise ValueError("Sistema RAG non inizializzato. Chiamare setup_qa_chain prima.")
        
        try:
            # Ottieni la memoria per questa sessione
            memory = self.memories[session_id]
            
            # Formatta la chat history come testo
            chat_vars = memory.load_memory_variables({})
            messages = chat_vars.get("chat_history", [])
            formatted_history = ""
            
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    formatted_history += f"USER: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    formatted_history += f"ASSISTANT: {msg.content}\n"
            
            # Analizza l'intento dell'utente
            is_bandi_question = self._analyze_user_intent(question)
            
            # Se è una domanda sui bandi, usa il retriever
            if is_bandi_question:
                # Esegui la query con il retriever
                result = self.qa_chain.invoke({
                    "question": question,
                    "chat_history": formatted_history
                })
                
                # Estrai informazioni sui documenti fonte
                source_docs = result.get("source_documents", [])
                sources = []
                
                for doc in source_docs:
                    source_info = {
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", 0),
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    }
                    sources.append(source_info)
            else:
                # Se è chat generale, usa solo il modello senza retriever
                chat_prompt = f"""
                Sei un assistente amichevole ed esperto nell'analisi di bandi pubblici e finanziamenti.
                
                CRONOLOGIA DELLA CONVERSAZIONE:
                {formatted_history}
                
                DOMANDA ATTUALE: {question}
                
                Rispondi in modo amichevole e professionale, senza cercare informazioni nei bandi poiché la domanda è di carattere generale.
                """
                
                response = self.llm.invoke(chat_prompt)
                result = {
                    "answer": response.content if hasattr(response, 'content') else str(response),
                    "source_documents": []
                }
                sources = []
            
            return {
                "answer": result["answer"],
                "sources": sources,
                "question": question,
                "chat_history": messages
            }
            
        except Exception as e:
            logger.error(f"Errore nella query per sessione {session_id}: {str(e)}")
            raise
    
    def clear_memory(self, session_id: str):
        """Pulisce la memoria per una sessione specifica"""
        if session_id in self.chat_histories:
            logger.info(f"Pulizia memoria per sessione {session_id}")
            self.chat_histories[session_id].clear()
    
    def save_memory(self, session_id: str) -> Dict[str, Any]:
        """Salva lo stato della memoria per una sessione"""
        if session_id in self.chat_histories:
            return {"messages": self.chat_histories[session_id].messages}
        return {"messages": []}
    
    def load_memory(self, session_id: str, memory_data: Dict[str, Any]):
        """Carica lo stato della memoria per una sessione"""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = ChatMessageHistory()
        
        if "messages" in memory_data:
            for msg in memory_data["messages"]:
                if isinstance(msg, (HumanMessage, AIMessage)):
                    self.chat_histories[session_id].add_message(msg)
    
    def get_memory_summary(self, session_id: str) -> Dict[str, Any]:
        """Ottiene un riepilogo della memoria della sessione"""
        if session_id in self.chat_histories:
            messages = self.chat_histories[session_id].messages
            return {
                "message_count": len(messages),
                "last_interaction": messages[-2:] if len(messages) >= 2 else [],
                "has_memory": bool(messages)
            }
        return {
            "message_count": 0,
            "last_interaction": [],
            "has_memory": False
        }
    
    def extract_structured_info(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Estrae informazioni strutturate dai bandi"""
        
        extraction_prompt = """
        Analizza il seguente testo di un bando e estrai le seguenti informazioni strutturate:
        
        Testo del bando:
        {text}
        
        Estrai e formatta le seguenti informazioni (se disponibili):
        1. Nome del bando
        2. Ente erogatore
        3. Scadenza per la presentazione
        4. Budget totale disponibile
        5. Importo massimo finanziabile per progetto
        6. Settori/ambiti di applicazione
        7. Requisiti principali
        8. Tipologia di beneficiari
        9. Percentuale di cofinanziamento
        10. Link o riferimenti al bando completo
        
        Risposta in formato JSON:
        """
        
        extracted_info = []
        
        for doc in documents:
            try:
                # Limita il testo per evitare token eccessivi
                text_chunk = doc.page_content[:3000]
                
                prompt = extraction_prompt.format(text=text_chunk)
                response = self.llm.invoke(prompt)
                
                # Aggiungi metadati del documento
                info = {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 0),
                    "extracted_data": response.content,
                    "file_path": doc.metadata.get("file_path", "")
                }
                
                extracted_info.append(info)
                
            except Exception as e:
                logger.error(f"Errore nell'estrazione per documento {doc.metadata.get('source', 'Unknown')}: {str(e)}")
                continue
        
        return extracted_info
    
    def search_by_project_idea(self, project_idea: str, vector_store) -> List[Dict[str, Any]]:
        """Cerca bandi compatibili con un'idea progettuale"""
        
        search_prompt = f"""
        Idea progettuale: {project_idea}
        
        Basandoti sui documenti disponibili, trova i bandi che potrebbero finanziare questa idea progettuale.
        
        Considera:
        1. Settori di applicazione compatibili
        2. Tipologia di attività finanziabili
        3. Requisiti che potrebbero essere soddisfatti
        4. Budget disponibile
        
        Per ogni bando rilevante, fornisci:
        - Nome del bando
        - Motivo della compatibilità
        - Requisiti principali
        - Scadenza (se disponibile)
        - Budget (se disponibile)
        """
        
        try:
            # Usa il retriever per trovare documenti rilevanti
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )
            
            relevant_docs = retriever.get_relevant_documents(project_idea)
            
            # Analizza i documenti trovati
            results = []
            for doc in relevant_docs:
                analysis_prompt = f"""
                Idea progettuale: {project_idea}
                
                Documento del bando:
                {doc.page_content[:2000]}
                
                Questo bando è compatibile con l'idea progettuale? Spiega perché sì o no.
                Se compatibile, fornisci dettagli su requisiti, scadenze e budget.
                """
                
                response = self.llm.invoke(analysis_prompt)
                
                result = {
                    "source": doc.metadata.get("source", "Unknown"),
                    "compatibility_analysis": response.content,
                    "document_preview": doc.page_content[:300] + "...",
                    "file_path": doc.metadata.get("file_path", "")
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Errore nella ricerca per idea progettuale: {str(e)}")
            raise
    
    def generate_summary_table(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Genera una tabella di sintesi per tutti i bandi"""
        
        summary_prompt = """
        Analizza il seguente documento di bando e crea una riga di sintesi con le seguenti colonne:
        
        Documento:
        {text}
        
        Estrai le seguenti informazioni (inserisci "N/A" se non disponibile):
        1. Nome Bando
        2. Ente Erogatore
        3. Scadenza (formato: gg/mm/aaaa o "N/A")
        4. Budget Totale (formato: € importo o "N/A")
        5. Importo Max per Progetto (formato: € importo o "N/A")
        6. Settori (breve descrizione)
        7. Beneficiari (tipologia)
        8. Cofinanziamento % (formato: xx% o "N/A")
        9. Stato (Aperto/Chiuso/N/A)
        10. Note (informazioni aggiuntive importanti)
        
        IMPORTANTE: Rispondi SOLO con il formato richiesto separato da "|", senza spiegazioni aggiuntive.
        Formato risposta: Nome|Ente|Scadenza|Budget|ImportoMax|Settori|Beneficiari|Cofinanziamento|Stato|Note
        """
        
        summary_data = []
        
        for doc in documents:
            try:
                text_chunk = doc.page_content[:2500]
                prompt = summary_prompt.format(text=text_chunk)
                response = self.llm.invoke(prompt)
                
                # Parse della risposta
                response_content = response.content if hasattr(response, 'content') else str(response)
                if isinstance(response_content, str):
                    parts = response_content.split("|")
                    if len(parts) >= 10:
                        row = {
                            "Nome Bando": parts[0].strip(),
                            "Ente Erogatore": parts[1].strip(),
                            "Scadenza": parts[2].strip(),
                            "Budget Totale": parts[3].strip(),
                            "Importo Max per Progetto": parts[4].strip(),
                            "Settori": parts[5].strip(),
                            "Beneficiari": parts[6].strip(),
                            "Cofinanziamento %": parts[7].strip(),
                            "Stato": parts[8].strip(),
                            "Note": parts[9].strip(),
                            "Fonte": doc.metadata.get("source", "Unknown"),
                            "Pagina": doc.metadata.get("page", 0)
                        }
                        summary_data.append(row)
                
            except Exception as e:
                logger.error(f"Errore nella creazione della sintesi per {doc.metadata.get('source', 'Unknown')}: {str(e)}")
                continue
        
        return summary_data 