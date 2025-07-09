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
from src.document_processor import EnhancedDocumentProcessor
from src.chat_manager import ChatManager
import logging
from unstructured.partition.pdf import partition_pdf
import streamlit as st  # Import necessario per utilizzare Streamlit

logger = logging.getLogger(__name__)

class RAGSystem:
    """Sistema RAG per interrogare i bandi"""
    
    def __init__(self):
        self.config = Config()
        self.document_processor = EnhancedDocumentProcessor()
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
    
    def query(self, question: str, session_id: str) -> Dict[str, Any]:
        """Esegue una query sul sistema RAG"""
        if not self.qa_chain:
            raise ValueError("Sistema RAG non inizializzato. Chiamare setup_qa_chain prima.")
        
        try:
            memory = self.memories.get(session_id)
            if not memory:
                raise ValueError(f"Memoria non trovata per sessione {session_id}")
            
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
                
                Se la domanda non riguarda i bandi caricati, rispondi semplicemente:
                L'informazione richiesta non è presente nei documenti caricati. Per favore, chiedi qualcosa relativo ai bandi disponibili.
                Rispondi in modo breve, chiaro e gentile. Se è solo un saluto o una conversazione frivola, rispondi cordialmente.
                Se ti viene chiesto, il tuo nome è Bandi Assistant.
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

    def _analyze_user_intent(self, question: str) -> bool:
        """Analizza l'intento dell'utente per capire se sta chiedendo informazioni sui bandi"""
        intent_prompt = """
        Analizza il seguente messaggio e determina se l'utente sta chiedendo informazioni specifiche sui bandi/documenti o sta solo chattando.
        
        Messaggio utente: {question}
        
        Rispondi SOLO con "True" se la domanda è sui bandi, "False" se è chat generale.
        """
        
        try:
            response = self.llm.invoke(intent_prompt.format(question=question))
            result = str(response.content if hasattr(response, 'content') else response).strip().lower()
            return result == "true"
        except Exception as e:
            logger.error(f"Errore nell'analisi dell'intento: {str(e)}")
            return True

    
    # Aggiungi questo metodo alla classe RAGSystem nel file rag_system.py

    def setup_qa_chain(self, vector_store, session_id: str, initial_chat_history: str = ""):
        """Configura la catena QA con supporto migliorato per contenuto Markdown"""
        
        qa_template_markdown = """
        Sei un assistente amichevole ed esperto nell'analisi di bandi pubblici e finanziamenti.
        Il tuo compito è aiutare l'utente a trovare e comprendere i bandi più adatti alle sue esigenze.
        
        IMPORTANTE: I documenti sono stati convertiti da PDF a Markdown per una migliore leggibilità.
        Utilizza questa strutturazione per fornire risposte più precise e ben formattate.

        Se l'informazione richiesta NON è presente nei documenti caricati, rispondi chiaramente:
        "L'informazione richiesta non è presente nei documenti caricati. Fai una domanda pertinente alla documentazione."
        NON inventare o allargare la risposta.
        
        CRONOLOGIA DELLA CONVERSAZIONE:
        {chat_history}
        
        CONTESTO SUI BANDI (formato Markdown):
        {context}
        
        DOMANDA ATTUALE: {question}
        
        ISTRUZIONI OBBLIGATORIE:
        1. Usa SEMPRE le informazioni dalla cronologia della conversazione per personalizzare le risposte
        2. Se l'utente ha fornito il suo nome in precedenza, usalo sempre
        3. Fai riferimento a domande e risposte precedenti quando rilevante
        4. Mantieni un tono amichevole e professionale
        5. Sfrutta la formattazione Markdown dei documenti per:
        - Identificare titoli e sezioni (# ## ###)
        - Riconoscere liste e tabelle
        - Evidenziare informazioni strutturate
        6. Quando citi informazioni da un bando, includi SEMPRE:
        - Il nome del bando
        - La sezione specifica (se presente in Markdown)
        - Il link al bando (se disponibile nei metadati)
        - La fonte specifica dell'informazione
        7. Formatta le tue risposte usando Markdown quando appropriato:
        - **Grassetto** per informazioni importanti
        - `Codice` per riferimenti specifici
        - > Citazioni per estratti dai bandi
        - Liste puntate per elenchi
        8. Se trovi tabelle o liste strutturate nei documenti Markdown, mantieni la formattazione
        9. NON cercare le informazioni sul web, usa solo i documenti forniti
        10. Se l'informazione richiesta NON è presente nei documenti caricati, rispondi chiaramente:
        "L'informazione richiesta non è presente nei documenti caricati. Fai una domanda pertinente alla documentazione."
        NON inventare o allargare la risposta.
        
        RISPOSTA (usa formattazione Markdown quando utile):
        """
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        if initial_chat_history:
            memory.chat_memory.add_ai_message(initial_chat_history)
        
        # Retriever migliorato per contenuto Markdown
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 6,  # Leggermente più documenti per Markdown
                "fetch_k": 12  # Più opzioni nella ricerca iniziale
            }
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template=qa_template_markdown,
                    input_variables=["context", "chat_history", "question"]
                )
            },
            return_source_documents=True,
            verbose=True
        )
        
        self.memories[session_id] = memory
        logger.info(f"Catena QA con supporto Markdown configurata per sessione {session_id}")

    def extract_markdown_structure(self, documents: List[Document]) -> Dict[str, Any]:
        """Estrae la struttura dai documenti Markdown per analisi migliore"""
        structure_info = {
            "total_documents": len(documents),
            "markdown_documents": 0,
            "has_tables": False,
            "has_headers": False,
            "has_lists": False,
            "sections": [],
            "conversion_methods": set()
        }
        
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata
            
            # Conta documenti Markdown
            if metadata.get('content_type') == 'markdown':
                structure_info["markdown_documents"] += 1
            
            # Rileva conversioni
            conv_method = metadata.get('conversion_method', 'unknown')
            structure_info["conversion_methods"].add(conv_method)
            
            # Analizza contenuto Markdown
            if content:
                # Headers
                if any(line.startswith('#') for line in content.split('\n')):
                    structure_info["has_headers"] = True
                    
                    # Estrai titoli principali
                    for line in content.split('\n'):
                        if line.startswith('# ') and len(line) > 2:
                            title = line[2:].strip()
                            if title and title not in structure_info["sections"]:
                                structure_info["sections"].append(title)
                
                # Liste
                if any(line.strip().startswith(('-', '*', '+')) for line in content.split('\n')):
                    structure_info["has_lists"] = True
                
                # Tabelle (format Markdown)
                if '|' in content and any('---' in line for line in content.split('\n')):
                    structure_info["has_tables"] = True
        
        structure_info["conversion_methods"] = list(structure_info["conversion_methods"])
        return structure_info

    def generate_enhanced_summary_table(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Genera tabella di sintesi sfruttando la struttura Markdown"""
        try:
            # Analizza struttura
            structure = self.extract_markdown_structure(documents)
            logger.info(f"Struttura documenti: {structure}")
            
            # Raggruppa i documenti per nome file
            docs_by_file = {}
            for doc in documents:
                file_name = doc.metadata.get('source', 'Unknown')
                if file_name not in docs_by_file:
                    docs_by_file[file_name] = []
                docs_by_file[file_name].append(doc)

            summary_data = []

            # Prompt migliorato per documenti Markdown
            extract_prompt_markdown = '''
            Sei un esperto analista di bandi pubblici. Il documento seguente è stato convertito da PDF a Markdown 
            per una migliore strutturazione. Sfrutta questa formattazione per estrarre informazioni precise.

            REGOLE IMPORTANTI:
            1. Usa la struttura Markdown (# titoli, ## sottotitoli, liste, tabelle) per navigare il documento
            2. Cerca informazioni nelle sezioni appropriate (es. "Requisiti", "Budget", "Scadenze")
            3. Se trovi tabelle in formato Markdown, estrapolane i dati
            4. NON rispondere MAI con "N/A" a meno che sia ASSOLUTAMENTE IMPOSSIBILE trovare l'informazione
            5. Sfrutta la formattazione per identificare liste di beneficiari, settori, etc.

            Documento Markdown del bando:
            {text}

            ESTRAI LE SEGUENTI INFORMAZIONI E RESTITUISCILE NEL SEGUENTE FORMATO:
            Nome Bando: [usa il titolo principale # o il nome del file]
            Ente Erogatore: [cerca in intestazioni o sezioni "Ente", "Organizzazione"]
            Scadenza: [cerca in sezioni "Scadenze", "Termini", tabelle con date - formato: gg/mm/aaaa]
            Budget Totale: [cerca "Budget", "Dotazione", "Risorse", tabelle finanziarie]
            Importo Max per Progetto: [cerca "Importo massimo", "Limite", tabelle con cifre]
            Settori: [cerca sezioni "Settori", "Ambiti", liste puntate con settori]
            Beneficiari: [cerca sezioni "Beneficiari", "Destinatari", liste di soggetti ammessi]
            Cofinanziamento %: [cerca percentuali, tabelle con cofinanziamento]
            Stato: [deduci da date e contesto]
            Note: [informazioni aggiuntive rilevanti]

            La risposta deve contenere SOLO i campi richiesti senza testo aggiuntivo.
            '''

            for file_name, file_docs in docs_by_file.items():
                try:
                    # Combina il testo mantenendo la struttura Markdown
                    full_text = ""
                    for doc in file_docs:
                        if doc.metadata.get('page'):
                            full_text += f"\n\n---\n**Pagina {doc.metadata['page']}**\n\n"
                        full_text += doc.page_content

                    # Limita la lunghezza ma cerca di mantenere sezioni complete
                    if len(full_text) > 50000:
                        # Cerca di tagliare a fine sezione
                        truncated = full_text[:50000]
                        last_section = truncated.rfind('\n## ')
                        if last_section > 30000:  # Se troviamo una sezione ragionevole
                            full_text = truncated[:last_section]
                        else:
                            full_text = truncated

                    # Estrai le informazioni
                    response = self.llm.invoke(
                        extract_prompt_markdown.format(text=full_text)
                    )

                    # Parse della risposta
                    info = self._parse_extraction_response_markdown(
                        str(response.content if hasattr(response, 'content') else response),
                        file_name
                    )

                    info['source'] = file_name
                    info['url'] = file_docs[0].metadata.get('url', 'N/A')
                    info['conversion_method'] = file_docs[0].metadata.get('conversion_method', 'unknown')

                    summary_data.append(info)

                except Exception as e:
                    logger.error(f"Errore nell'elaborazione Markdown del documento {file_name}: {str(e)}")
                    summary_data.append(self._create_fallback_entry(file_name, str(e)))

            return summary_data

        except Exception as e:
            logger.error(f"Errore nella generazione della tabella di sintesi Markdown: {str(e)}")
            raise

    def _parse_extraction_response_markdown(self, response_text: str, file_name: str) -> Dict[str, str]:
        """Parsing migliorato per risposte da documenti Markdown"""
        info = {}
        
        # Campi richiesti con valori di default migliorati
        required_fields = {
            'Nome Bando': os.path.splitext(os.path.basename(file_name))[0],
            'Ente Erogatore': 'Da verificare nel documento',
            'Scadenza': 'Da verificare nel documento',
            'Budget Totale': 'Da verificare nel documento',
            'Importo Max per Progetto': 'Da verificare nel documento',
            'Settori': 'Da verificare nel documento',
            'Beneficiari': 'Da verificare nel documento',
            'Cofinanziamento %': 'Da verificare nel documento',
            'Stato': 'Da verificare nel documento',
            'Note': 'Documento convertito da Markdown'
        }
        
        # Parsing delle righe con gestione migliorata
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line and not line.startswith('#'):  # Evita headers Markdown
                parts = line.split(':', 1)
                if len(parts) == 2:
                    field = parts[0].strip()
                    value = parts[1].strip()
                    
                    # Pulisci il valore
                    # Rimuovi caratteri markdown extra
                    value = value.replace('**', '').replace('*', '').replace('`', '')
                    
                    # Rimuovi parentesi quadre se presenti
                    if value.startswith('[') and value.endswith(']'):
                        value = value[1:-1].strip()
                    
                    # Rimuovi virgolette
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1].strip()
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1].strip()
                    
                    # Applica il valore se non vuoto e significativo
                    if (field in required_fields and value and 
                        value.lower() not in ['n/a', 'non disponibile', 'nd', 'da definire', 'tbd']):
                        info[field] = value
        
        # Applica valori di default per campi mancanti
        for field, default_value in required_fields.items():
            if field not in info:
                info[field] = default_value
        
        return info

    def analyze_markdown_content_quality(self, documents: List[Document]) -> Dict[str, Any]:
        """Analizza la qualità della conversione Markdown"""
        analysis = {
            "total_documents": len(documents),
            "markdown_converted": 0,
            "average_length": 0,
            "structure_quality": {
                "has_headers": 0,
                "has_tables": 0,
                "has_lists": 0,
                "well_structured": 0
            },
            "conversion_methods": {},
            "potential_issues": []
        }
        
        total_length = 0
        
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata
            
            total_length += len(content)
            
            # Conta documenti convertiti in Markdown
            if metadata.get('content_type') == 'markdown':
                analysis["markdown_converted"] += 1
            
            # Analizza metodo di conversione
            conv_method = metadata.get('conversion_method', 'unknown')
            analysis["conversion_methods"][conv_method] = analysis["conversion_methods"].get(conv_method, 0) + 1
            
            # Analizza struttura
            if content:
                lines = content.split('\n')
                
                # Headers
                if any(line.strip().startswith('#') for line in lines):
                    analysis["structure_quality"]["has_headers"] += 1
                
                # Tabelle Markdown
                if any('|' in line and '---' in content for line in lines):
                    analysis["structure_quality"]["has_tables"] += 1
                
                # Liste
                if any(line.strip().startswith(('-', '*', '+', '1.')) for line in lines):
                    analysis["structure_quality"]["has_lists"] += 1
                
                # Documento ben strutturato (ha headers + liste o tabelle)
                has_structure = (
                    any(line.strip().startswith('#') for line in lines) and
                    (any(line.strip().startswith(('-', '*', '+')) for line in lines) or
                    any('|' in line for line in lines))
                )
                if has_structure:
                    analysis["structure_quality"]["well_structured"] += 1
            
            # Identifica potenziali problemi
            if len(content) < 100:
                analysis["potential_issues"].append(f"Documento troppo corto: {metadata.get('source', 'unknown')}")
            
            if content.count('\n') < 5:
                analysis["potential_issues"].append(f"Poca strutturazione: {metadata.get('source', 'unknown')}")
        
        # Calcola medie
        if len(documents) > 0:
            analysis["average_length"] = total_length / len(documents)
        
        # Calcola percentuali
        for key in analysis["structure_quality"]:
            analysis["structure_quality"][key] = {
                "count": analysis["structure_quality"][key],
                "percentage": (analysis["structure_quality"][key] / len(documents)) * 100 if len(documents) > 0 else 0
            }
        
        return analysis

    def get_markdown_search_suggestions(self, query: str, documents: List[Document]) -> List[str]:
        """Genera suggerimenti di ricerca basati sulla struttura Markdown"""
        suggestions = []
        
        # Analizza i documenti per trovare sezioni comuni
        common_sections = set()
        common_topics = set()
        
        for doc in documents:
            content = doc.page_content
            if content:
                lines = content.split('\n')
                
                # Estrai headers
                for line in lines:
                    if line.strip().startswith('#'):
                        header = line.strip().lstrip('#').strip().lower()
                        if len(header) > 3:  # Evita headers troppo corti
                            common_sections.add(header)
                
                # Estrai parole chiave da liste
                for line in lines:
                    if line.strip().startswith(('-', '*', '+')):
                        item = line.strip().lstrip('-*+').strip().lower()
                        if len(item) > 5 and len(item) < 50:
                            common_topics.add(item)
        
        # Genera suggerimenti basati sulla query
        query_lower = query.lower()
        
        # Suggerimenti per sezioni
        relevant_sections = [s for s in common_sections if any(word in s for word in query_lower.split())]
        for section in sorted(relevant_sections)[:3]:
            suggestions.append(f"Informazioni sulla sezione '{section.title()}'")
        
        # Suggerimenti generici utili
        generic_suggestions = [
            "Quali sono i requisiti per partecipare?",
            "Qual è la scadenza del bando?",
            "Quanto budget è disponibile?",
            "Chi sono i beneficiari ammessi?",
            "Come si presenta la domanda?",
            "Quali documenti sono richiesti?",
            "Qual è la percentuale di cofinanziamento?"
        ]

        # Aggiungi suggerimenti generici alla lista dei suggerimenti
        suggestions.extend(generic_suggestions)
        
        return suggestions[:5]

    # Aggiungi anche questo metodo per debugging e monitoraggio
    def debug_markdown_processing(self, documents: List[Document]) -> str:
        """Genera un report di debug per il processamento Markdown"""
        
        report = "# 🔍 Report Debug Processamento Markdown\n\n"
        
        # Analisi generale
        analysis = self.analyze_markdown_content_quality(documents)
        
        report += f"## 📊 Statistiche Generali\n\n"
        report += f"- **Documenti totali**: {analysis['total_documents']}\n"
        report += f"- **Convertiti in Markdown**: {analysis['markdown_converted']}\n"
        report += f"- **Lunghezza media**: {analysis['average_length']:.0f} caratteri\n\n"
        
        # Metodi di conversione
        report += f"## 🔄 Metodi di Conversione\n\n"
        for method, count in analysis['conversion_methods'].items():
            report += f"- **{method}**: {count} documenti\n"
        
        # Qualità struttura
        report += f"\n## 🏗️ Qualità Struttura\n\n"
        for feature, data in analysis['structure_quality'].items():
            report += f"- **{feature.replace('_', ' ').title()}**: {data['count']} documenti ({data['percentage']:.1f}%)\n"
        
        # Problemi potenziali
        if analysis['potential_issues']:
            report += f"\n## ⚠️ Problemi Potenziali\n\n"
            for issue in analysis['potential_issues'][:5]:  # Primi 5 problemi
                report += f"- {issue}\n"
        
        # Esempi di contenuto
        report += f"\n## 📝 Esempi di Contenuto\n\n"
        markdown_docs = [doc for doc in documents if doc.metadata.get('content_type') == 'markdown']
        
        if markdown_docs:
            sample_doc = markdown_docs[0]
            preview = sample_doc.page_content[:500]
            report += f"**File**: {sample_doc.metadata.get('source', 'Unknown')}\n\n"
            report += f"```markdown\n{preview}...\n```\n\n"
        
        # Raccomandazioni
        report += f"## 💡 Raccomandazioni\n\n"
        
        markdown_ratio = analysis['markdown_converted'] / analysis['total_documents'] if analysis['total_documents'] > 0 else 0
        
        if markdown_ratio < 0.8:
            report += "- ⚠️ Molti documenti non sono stati convertiti in Markdown. Verifica la configurazione.\n"
        
        if analysis['structure_quality']['well_structured']['percentage'] < 50:
            report += "- 📋 Pochi documenti hanno una buona struttura. Considera di migliorare la conversione.\n"
        
        if analysis['structure_quality']['has_tables']['percentage'] > 20:
            report += "- 📊 I documenti contengono tabelle. La conversione Markdown dovrebbe preservarle meglio.\n"
        
        if not analysis['potential_issues']:
            report += "- ✅ Nessun problema significativo rilevato!\n"
        
        return report
    
    # Aggiungi questi metodi alla classe RAGSystem nel file src/rag_system.py

    def generate_summary_table(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Genera tabella di sintesi con struttura Excel specifica"""
        try:
            docs_by_file = {}
            for doc in documents:
                file_name = doc.metadata.get('source', 'Unknown')
                if file_name not in docs_by_file:
                    docs_by_file[file_name] = []
                docs_by_file[file_name].append(doc)

            summary_data = []
            extract_prompt = '''
            Estrai informazioni dal bando seguendo ESATTAMENTE questa struttura:

            Ente erogatore: [Regione, Ministero, Comune, etc.]
            Titolo dell'avviso: [Titolo ufficiale del bando]
            Descrizione aggiuntiva: [Breve descrizione obiettivi]
            Beneficiari: [Chi può partecipare]
            Apertura: [Data apertura - formato gg/mm/aaaa]
            Chiusura: [Data scadenza - formato gg/mm/aaaa]
            Dotazione finanziaria: [Budget totale]
            Contributo: [Importo massimo per progetto]
            Note: [Informazioni aggiuntive]
            Link: [URL al pdf]
            Key Words: [Parole chiave separate da virgole]
            Aperto (si/no): [Si se ancora aperto, No se chiuso]

            Testo del bando:
            {text}

            Rispondi SOLO nel formato richiesto.
            '''

            for file_name, file_docs in docs_by_file.items():
                try:
                    full_text = "\n".join([doc.page_content for doc in file_docs])
                    response = self.llm.invoke(extract_prompt.format(text=full_text[:50000]))
                    
                    info = self._parse_excel_response(
                        str(response.content if hasattr(response, 'content') else response),
                        file_name
                    )
                    summary_data.append(info)

                except Exception as e:
                    logger.error(f"Errore elaborazione {file_name}: {str(e)}")
                    summary_data.append(self._create_excel_fallback(file_name, str(e)))

            return summary_data

        except Exception as e:
            logger.error(f"Errore generazione sintesi: {str(e)}")
            raise

    def _parse_excel_response(self, response_text: str, file_name: str) -> Dict[str, str]:
        """Parse risposta seguendo struttura Excel"""
        from datetime import datetime
        import re
        
        fields = [
            'Ente erogatore', 'Titolo dell\'avviso', 'Descrizione aggiuntiva',
            'Beneficiari', 'Apertura', 'Chiusura', 'Dotazione finanziaria',
            'Contributo', 'Note', 'Link', 'Key Words', 'Aperto (si/no)'
        ]
        
        info = {}
        lines = response_text.split('\n')
        
        for line in lines:
            if ':' in line and not line.startswith('#'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    field = parts[0].strip()
                    value = parts[1].strip()
                    
                    # Pulisci valore
                    value = value.replace('**', '').replace('*', '').replace('`', '')
                    if value.startswith('[') and value.endswith(']'):
                        value = value[1:-1].strip()
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1].strip()
                    
                    if field in fields and value and value.lower() not in ['n/a', 'non disponibile', 'nd']:
                        info[field] = value
        
        # Applica default per campi mancanti
        defaults = {
            'Ente erogatore': 'Regione Lombardia',
            'Titolo dell\'avviso': os.path.splitext(os.path.basename(file_name))[0],
            'Descrizione aggiuntiva': 'Da verificare',
            'Beneficiari': 'Da verificare',
            'Apertura': 'Da verificare',
            'Chiusura': 'Da verificare',
            'Dotazione finanziaria': 'Da verificare',
            'Contributo': 'Da verificare',
            'Note': 'Da verificare',
            'Link': 'Da verificare',
            'Key Words': 'bando, finanziamento, lombardia',
            'Aperto (si/no)': 'Da verificare'
        }
        
        for field in fields:
            if field not in info:
                info[field] = defaults[field]
        
        # Valida date e determina se aperto
        info = self._validate_dates_and_status(info)
        
        return info

    def _validate_dates_and_status(self, info: Dict[str, str]) -> Dict[str, str]:
        """Valida date e determina stato aperto/chiuso"""
        from datetime import datetime
        import re
        
        # Valida formato date
        date_fields = ['Apertura', 'Chiusura']
        for field in date_fields:
            if field in info and info[field] != 'Da verificare':
                info[field] = self._format_date(info[field])
        
        # Determina se aperto basandosi sulla data di chiusura
        chiusura = info.get('Chiusura', 'Da verificare')
        if chiusura != 'Da verificare':
            try:
                match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', chiusura)
                if match:
                    day, month, year = map(int, match.groups())
                    closing_date = datetime(year, month, day)
                    today = datetime.now()
                    
                    info['Aperto (si/no)'] = 'Si' if closing_date >= today else 'No'
                else:
                    info['Aperto (si/no)'] = 'Da verificare'
            except:
                info['Aperto (si/no)'] = 'Da verificare'
        else:
            # Se non c'è data di chiusura, mantieni il valore esistente o 'Da verificare'
            if info.get('Aperto (si/no)', '').lower() not in ['si', 'no']:
                info['Aperto (si/no)'] = 'Da verificare'
        
        return info

    def _format_date(self, date_str: str) -> str:
        """Formatta data in gg/mm/aaaa"""
        import re
        from datetime import datetime
        
        patterns = [
            r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})',
            r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2})',
            r'(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    if len(match.group(3)) == 4:
                        if int(match.group(1)) > 12:  # gg/mm/aaaa
                            day, month, year = match.groups()
                        else:  # Potrebbe essere aaaa/mm/gg
                            if int(match.group(1)) > 31:
                                year, month, day = match.groups()
                            else:
                                day, month, year = match.groups()
                    else:  # Anno a 2 cifre
                        day, month, year = match.groups()
                        year = f"20{year}" if int(year) < 50 else f"19{year}"
                    
                    datetime(int(year), int(month), int(day))
                    return f"{int(day):02d}/{int(month):02d}/{year}"
                except ValueError:
                    continue
        
        return 'Da verificare'

    def _create_excel_fallback(self, file_name: str, error_msg: str) -> Dict[str, str]:
        """Crea entry di fallback per struttura Excel"""
        return {
            'Ente erogatore': 'Regione Lombardia',
            'Titolo dell\'avviso': os.path.splitext(os.path.basename(file_name))[0],
            'Descrizione aggiuntiva': 'Da verificare',
            'Beneficiari': 'Da verificare',
            'Apertura': 'Da verificare',
            'Chiusura': 'Da verificare',
            'Dotazione finanziaria': 'Da verificare',
            'Contributo': 'Da verificare',
            'Note': f'Errore: {error_msg[:50]}',
            'Link': 'Da verificare',
            'Key Words': 'bando, finanziamento, lombardia',
            'Aperto (si/no)': 'Da verificare'
        }


    def generate_enhanced_summary_table(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Versione enhanced per documenti Markdown"""
        try:
            docs_by_file = {}
            for doc in documents:
                file_name = doc.metadata.get('source', 'Unknown')
                if file_name not in docs_by_file:
                    docs_by_file[file_name] = []
                docs_by_file[file_name].append(doc)

            summary_data = []
            extract_prompt = '''
            Estrai dal documento Markdown seguendo questa struttura:

            Ente erogatore: [cerca in header o sezioni ente]
            Titolo dell'avviso: [titolo principale # o il nome ufficiale]
            Descrizione aggiuntiva: [obiettivi, finalità]
            Beneficiari: [sezioni beneficiari, destinatari]
            Apertura: [data apertura - gg/mm/aaaa]
            Chiusura: [data scadenza - gg/mm/aaaa]
            Dotazione finanziaria: [budget totale]
            Contributo: [importo max per progetto]
            Note: [informazioni aggiuntive]
            Link: [URL del pdf]
            Key Words: [parole chiave separate da virgole]
            Aperto (si/no): [Si se aperto, No se chiuso]

            Documento:
            {text}

            Rispondi nel formato richiesto.
            '''

            for file_name, file_docs in docs_by_file.items():
                try:
                    full_text = ""
                    for doc in file_docs:
                        if doc.metadata.get('page'):
                            full_text += f"\n---Pagina {doc.metadata['page']}---\n"
                        full_text += doc.page_content

                    if len(full_text) > 60000:
                        full_text = full_text[:60000]

                    response = self.llm.invoke(extract_prompt.format(text=full_text))
                    info = self._parse_excel_response(
                        str(response.content if hasattr(response, 'content') else response),
                        file_name
                    )
                    summary_data.append(info)

                except Exception as e:
                    logger.error(f"Errore Markdown {file_name}: {str(e)}")
                    summary_data.append(self._create_excel_fallback(file_name, str(e)))

            return summary_data

        except Exception as e:
            logger.error(f"Errore enhanced sintesi: {str(e)}")
            raise
    
    def _parse_extraction_response_markdown(self, response_text: str, file_name: str) -> Dict[str, str]:
        """Parsing migliorato per risposte da documenti Markdown"""
        info = {}
        
        # Campi richiesti con valori di default migliorati
        required_fields = {
            'Nome Bando': os.path.splitext(os.path.basename(file_name))[0],
            'Ente Erogatore': 'Da verificare nel documento',
            'Scadenza': 'Da verificare nel documento',
            'Budget Totale': 'Da verificare nel documento',
            'Importo Max per Progetto': 'Da verificare nel documento',
            'Settori': 'Da verificare nel documento',
            'Beneficiari': 'Da verificare nel documento',
            'Cofinanziamento %': 'Da verificare nel documento',
            'Stato': 'Da verificare nel documento',
            'Note': 'Documento convertito da Markdown'
        }
        
        # Parsing delle righe con gestione migliorata
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line and not line.startswith('#'):  # Evita headers Markdown
                parts = line.split(':', 1)
                if len(parts) == 2:
                    field = parts[0].strip()
                    value = parts[1].strip()
                    
                    # Pulisci il valore
                    # Rimuovi caratteri markdown extra
                    value = value.replace('**', '').replace('*', '').replace('`', '')
                    
                    # Rimuovi parentesi quadre se presenti
                    if value.startswith('[') and value.endswith(']'):
                        value = value[1:-1].strip()
                    
                    # Rimuovi virgolette
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1].strip()
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1].strip()
                    
                    # Applica il valore se non vuoto e significativo
                    if (field in required_fields and value and 
                        value.lower() not in ['n/a', 'non disponibile', 'nd', 'da definire', 'tbd']):
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