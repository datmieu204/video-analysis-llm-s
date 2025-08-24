# backend/app/agents/chatbot.py

import logging
from typing import List, Union, Dict, Any

from backend.app.core.multi_llms import MultiLLMs
from backend.app.core.embeddings import VectorStore
from backend.app.utils.config import OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Chatbot:
    def __init__(self, **kwargs):
        self.multi_llms = MultiLLMs(api_keys=OPENAI_API_KEY, model="gpt-4o-mini", **kwargs)
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_id = None

        try:
            self.vector_store = VectorStore()
            logger.info("Vector store initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            self.vector_store = None

    def set_session_id(self, session_id: str):
        self.session_id = session_id
        logger.info(f"Session ID set to: {session_id}")
    
    def get_session_id(self) -> str:
        return self.session_id

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        return self.conversation_history

    def clear_conversation_history(self):
        self.conversation_history = []
        logger.info("Conversation history cleared.")

    def add_to_conversation_history(self, question: str, response: Union[str, List[str]], response_type: str = "standard"):
        response_text = response[0] if isinstance(response, list) and response else str(response)
        
        interaction = {
            "session_id": self.session_id,
            "question": question,
            "response": response_text,
            "response_type": response_type
        }
        self.conversation_history.append(interaction)
        logger.info(f"Added to conversation history: Q: {question[:50]}...")

    def build_context_from_history(self, max_history: int = 5) -> str:
        recent_history = self.conversation_history[-max_history:]
        context_parts = []

        for interaction in recent_history:
            context_parts.append(f"User: {interaction['question']}")
            context_parts.append(f"Assistant: {interaction['response']}")
                
        return "\n".join(context_parts)

    def ask_chatbot(self, question: str) -> str:
        response = self.multi_llms.get_llm_chatbot().invoke(question)
        
        # Extract content from response object
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict) and 'content' in response:
            return response['content']
        elif isinstance(response, list) and response:
            first_response = response[0]
            if hasattr(first_response, 'content'):
                return first_response.content
            elif isinstance(first_response, dict) and 'content' in first_response:
                return first_response['content']
            return str(first_response)
        else:
            return str(response)
    
    def ask(self, question: str, include_history: bool = False) -> str:
        try:
            if include_history and self.conversation_history:
                context = self.build_context_from_history()
                full_question = f"{context}\nUser: {question}"
            else:
                full_question = f"User: {question}"

            response = self.ask_chatbot(full_question)
            self.add_to_conversation_history(question=question, response=response)
            return response
        except Exception as e:
            logger.error(f"Error occurred while asking chatbot: {e}")
            return "I'm sorry, I encountered an error processing your request."
            
    def get_relevant_context(self, query: str) -> str:
        if not self.vector_store:
            logger.error("Vector store is not initialized.")
            return "I don't know"

        try: 
            relevant_docs = self.vector_store.search_similar_content(query)
            if not relevant_docs:
                return "I don't know"

            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            return context
        except Exception as e:
            logger.error(f"Error occurred while getting relevant context: {e}")
            return "I'm sorry, I encountered an error processing your request."
        
    def add_transcript_to_vectorstore(self, transcript_id: str, transcript_text: str) -> bool:
        if not self.vector_store:
            logger.error("Vector store is not initialized.")
            return False

        try:
            self.vector_store.add_transcript(transcript_id, transcript_text)
            logger.info(f"Transcript ID {transcript_id} added to vector store.")
            return True
        except Exception as e:
            logger.error(f"Error adding transcript to vector store: {e}")
            return False

    def ask_with_rag(self, question: str, include_history: bool = False) -> str:
        try:
            context = self.get_relevant_context(question)

            context_parts = []
            if include_history and self.conversation_history:
                history_context = self.build_context_from_history()
                context_parts.append(f"Previous conversation:\n{history_context}")

            if context:
                context_parts.append(f"Relevant information from transcript:\n{context}")
                system_prompt = (
                    "You are a helpful assistant that answers questions based on video transcripts and conversation context. "
                    "Use the provided relevant information to answer the user's question accurately. "
                    "If the information doesn't contain the answer, say so clearly."
                )
            else:
                system_prompt = (
                    "You are a helpful assistant. Answer the user's question based on your knowledge and any provided context."
                )

            full_context = "\n\n".join(context_parts) if context_parts else ""
            full_question = f"{system_prompt}\n\n{full_context}\nUser: {question}"

            response = self.ask_chatbot(full_question)
            self.add_to_conversation_history(question=question, response=response, response_type="rag")
            return response
        
        except Exception as e:
            logger.error(f"Error occurred while asking with RAG: {e}")
            return "I'm sorry, I encountered an error processing your request."