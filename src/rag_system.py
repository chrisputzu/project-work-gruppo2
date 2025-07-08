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
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)

class RAGSystem:
    """Sistema RAG per interrogare i bandi"""
    
    def __init__(self):
        self.config = Config()
        self.document_processor = DocumentProcessor()
        self.llm = self._get_llm()
        self.qa_chain = None
        self.chat_histories: Dict[str, ChatMessageHistory] = {}
        self.memories: Dict[str, ConversationBufferMemory] = {}

    
    
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
        6. Quando citi informazioni da un bando, includi SEMPRE:
           - Il nome del bando
           - Il link al bando (se disponibile nei metadati)
           - La fonte specifica dell'informazione
        7. Formatta i link usando la sintassi markdown: [Nome Bando](URL)
        
        RISPOSTA:
        """
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        if initial_chat_history:
            memory.chat_memory.add_ai_message(initial_chat_history)
        
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
        
        self.memories[session_id] = memory
        logger.info(f"Catena QA configurata con successo per sessione {session_id}")
    
    def _analyze_user_intent(self, question: str) -> bool:
        """Analizza l'intento dell'utente per capire se sta chiedendo informazioni sui bandi"""
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
            response = self.llm.invoke(intent_prompt.format(question=question))
            result = str(response.content if hasattr(response, 'content') else response).strip().lower()
            return result == "true"
        except Exception as e:
            logger.error(f"Errore nell'analisi dell'intento: {str(e)}")
            return True
    
    def query(self, question: str, session_id: str) -> Dict[str, Any]:
        """Esegue una query sul sistema RAG"""
        if not self.qa_chain:
            raise ValueError("Sistema RAG non inizializzato. Chiamare setup_qa_chain prima.")
        
        try:
            memory = self.memories[session_id]
            chat_vars = memory.load_memory_variables({})
            messages = chat_vars.get("chat_history", [])
            formatted_history = ""
            
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    formatted_history += f"USER: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    formatted_history += f"ASSISTANT: {msg.content}\n"
            
            is_bandi_question = self._analyze_user_intent(question)
            
            if is_bandi_question:
                result = self.qa_chain.invoke({
                    "question": question,
                    "chat_history": formatted_history
                })
                
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
    
    def search_by_project_idea(self, project_idea: str, vector_store) -> List[Dict[str, Any]]:
        """Cerca bandi compatibili con un'idea progettuale"""
        try:
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )
            
            relevant_docs = retriever.get_relevant_documents(project_idea)
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
        """Genera una tabella di sintesi dei bandi, includendo le tabelle estratte dai PDF tramite unstructured.io"""
        from unstructured.partition.pdf import partition_pdf
    
        def extract_tables_from_pdf(file_path):
            """Estrae le tabelle da un PDF usando unstructured.io e restituisce una stringa markdown."""
            try:
                elements = partition_pdf(filename=file_path, strategy="hi_res")
                tables = [el for el in elements if el.category == "Table"]
                table_texts = []
                for table in tables:
                    # Preferisci HTML se disponibile, altrimenti testo semplice
                    if hasattr(table, "text_as_html"):
                        table_texts.append(table.text_as_html)
                    else:
                        table_texts.append(str(table))
                if table_texts:
                    return "\n\nTABELLE ESTRATTE DAL DOCUMENTO:\n" + "\n\n".join(table_texts)
                return ""
            except Exception as e:
                logger.error(f"Errore nell'estrazione delle tabelle da {file_path}: {str(e)}")
                return ""
    
        try:
            # Raggruppa i documenti per nome file
            docs_by_file = {}
            for doc in documents:
                file_name = doc.metadata.get('source', 'Unknown')
                if file_name not in docs_by_file:
                    docs_by_file[file_name] = []
                docs_by_file[file_name].append(doc)
    
            summary_data = []
    
            # Prompt per l'estrazione delle informazioni
            extract_prompt = '''
            Sei un esperto analista di bandi pubblici. Il tuo compito è estrarre informazioni precise dal seguente bando.
            DEVI TROVARE LE INFORMAZIONI RICHIESTE. Se non sono esplicitamente presenti, cerca di dedurle dal contesto.
    
            REGOLE IMPORTANTI:
            1. NON rispondere MAI con "N/A" a meno che sia ASSOLUTAMENTE IMPOSSIBILE trovare o dedurre l'informazione
            2. Cerca le informazioni in tutto il testo, non fermarti alla prima pagina
            3. Se trovi più valori possibili, scegli il più recente o il più rilevante
            4. Usa il nome del file come riferimento se non trovi un titolo esplicito
            5. Cerca di dedurre lo stato del bando dalle date o dal contesto
    
            Testo del bando:
            {text}
    
            ESTRAI LE SEGUENTI INFORMAZIONI E RESTITUISCILE NEL SEGUENTE FORMATO, LA RISPOSTA DOVRA' 
            CONTENERE SOLO I CAMPI RICHIESTI SENZA TESTO AGGIUNTIVO:
            Nome Bando: [OBBLIGATORIO - usa il titolo ufficiale o il nome del file se non lo trovi]
            Ente Erogatore: [OBBLIGATORIO - cerca riferimenti a Regione, Ministero, o altri enti]
            Scadenza: [cerca date di scadenza, termini di presentazione - formato: gg/mm/aaaa]
            Budget Totale: [cerca riferimenti a dotazione finanziaria, risorse disponibili, budget]
            Importo Max per Progetto: [cerca limiti di finanziamento, importo massimo concedibile]
            Settori: [cerca settori di intervento, ambiti, aree tematiche]
            Beneficiari: [OBBLIGATORIO - cerca soggetti ammissibili, destinatari, beneficiari]
            Cofinanziamento %: [cerca percentuale di cofinanziamento richiesto]
            Stato: [deduci se Aperto/Chiuso dalle date o dal contesto]
            Note: [inserisci informazioni importanti non coperte sopra]
    
            RICORDA: Il tuo obiettivo è fornire una sintesi UTILE. Evita "N/A" il più possibile.
            Rispondi SOLO nel formato richiesto, senza spiegazioni o testo aggiuntivo.
            '''
    
            for file_name, file_docs in docs_by_file.items():
                try:
                    # Estrai tabelle se PDF
                    table_text = ""
                    if file_name.lower().endswith(".pdf"):
                        file_path = file_docs[0].metadata.get("file_path", file_name)
                        table_text = extract_tables_from_pdf(file_path)
    
                    # Combina il testo di tutte le pagine
                    full_text = "\n".join([doc.page_content for doc in file_docs])
                    if table_text:
                        full_text += "\n\n" + table_text
    
                    # Estrai le informazioni
                    response = self.llm.invoke(
                        extract_prompt.format(text=full_text[:40000])  # Limite per contesto
                    )
    
                    # Estrai le informazioni dalla risposta
                    info = {}
                    try:
                        response_text = str(response.content if hasattr(response, 'content') else response)
                        for line in response_text.split('\n'):
                            if not isinstance(line, str):
                                continue
                            line = str(line).strip()
                            if not line or ':' not in line:
                                continue
                            parts = line.split(':', 1)
                            if len(parts) != 2:
                                continue
                            field = str(parts[0]).strip()
                            value = str(parts[1]).strip()
                            if value.startswith('[') and value.endswith(']'):
                                value = value[1:-1].strip()
                            if value and value.lower() != 'n/a':
                                info[field] = value
                            else:
                                if field == 'Nome Bando':
                                    info[field] = os.path.splitext(os.path.basename(file_name))[0]
                                elif field == 'Ente Erogatore':
                                    info[field] = 'Regione Lombardia'
                                elif field == 'Beneficiari':
                                    info[field] = 'Da verificare nel bando'
                                else:
                                    info[field] = 'N/A'
                    except Exception as e:
                        logger.error(f"Errore nel parsing della risposta: {str(e)}")
                        info = {}
    
                    required_fields = [
                        'Nome Bando', 'Ente Erogatore', 'Scadenza', 'Budget Totale',
                        'Importo Max per Progetto', 'Settori', 'Beneficiari',
                        'Cofinanziamento %', 'Stato', 'Note'
                    ]
                    for field in required_fields:
                        if field not in info:
                            if field == 'Nome Bando':
                                info[field] = os.path.splitext(os.path.basename(file_name))[0]
                            elif field == 'Ente Erogatore':
                                info[field] = 'Regione Lombardia'
                            elif field == 'Beneficiari':
                                info[field] = 'Da verificare nel bando'
                            else:
                                info[field] = 'N/A'
    
                    info['source'] = file_name
                    info['url'] = file_docs[0].metadata.get('url', 'N/A')
    
                    summary_data.append(info)
    
                except Exception as e:
                    logger.error(f"Errore nell'elaborazione del documento {file_name}: {str(e)}")
                    summary_data.append({
                        'Nome Bando': os.path.splitext(os.path.basename(file_name))[0],
                        'Ente Erogatore': 'Regione Lombardia',
                        'Scadenza': 'N/A',
                        'Budget Totale': 'N/A',
                        'Importo Max per Progetto': 'N/A',
                        'Settori': 'N/A',
                        'Beneficiari': 'Da verificare nel bando',
                        'Cofinanziamento %': 'N/A',
                        'Stato': 'N/A',
                        'Note': f'Errore nell\'elaborazione: {str(e)}',
                        'source': file_name,
                        'url': 'N/A'
                    })
    
            return summary_data
    
        except Exception as e:
            logger.error(f"Errore nella generazione della tabella di sintesi: {str(e)}")
            raise
    
    def _parse_extraction_response(self, response_text: str, file_name: str) -> Dict[str, str]:
        """Parsing migliorato della risposta di estrazione"""
        info = {}
        
        # Campi richiesti con valori di default
        required_fields = {
            'Nome Bando': os.path.splitext(os.path.basename(file_name))[0],
            'Ente Erogatore': 'Da verificare',
            'Scadenza': 'Da verificare',
            'Budget Totale': 'Da verificare',
            'Importo Max per Progetto': 'Da verificare',
            'Settori': 'Da verificare',
            'Beneficiari': 'Da verificare',
            'Cofinanziamento %': 'Da verificare',
            'Stato': 'Da verificare',
            'Note': 'Da verificare'
        }
        
        # Parsing delle righe
        for line in response_text.split('\n'):
            line = line.strip()
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    field = parts[0].strip()
                    value = parts[1].strip()
                    
                    # Rimuovi caratteri extra
                    if value.startswith('[') and value.endswith(']'):
                        value = value[1:-1].strip()
                    
                    # Applica il valore se non vuoto
                    if field in required_fields and value and value.lower() not in ['n/a', 'non disponibile', 'nd']:
                        info[field] = value
        
        # Applica valori di default per campi mancanti
        for field, default_value in required_fields.items():
            if field not in info:
                info[field] = default_value
        
        return info
    
    def _create_fallback_entry(self, file_name: str, error_msg: str) -> Dict[str, str]:
        """Crea una voce di fallback in caso di errore"""
        return {
            'Nome Bando': os.path.splitext(os.path.basename(file_name))[0],
            'Ente Erogatore': 'Da verificare',
            'Scadenza': 'Da verificare',
            'Budget Totale': 'Da verificare',
            'Importo Max per Progetto': 'Da verificare',
            'Settori': 'Da verificare',
            'Beneficiari': 'Da verificare',
            'Cofinanziamento %': 'Da verificare',
            'Stato': 'Da verificare',
            'Note': f'Errore elaborazione: {error_msg[:100]}',
            'source': file_name,
            'url': 'Da verificare'
        }
