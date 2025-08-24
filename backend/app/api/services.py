#backend/app/api/services.py

import os
import uuid
import logging
import tempfile

from typing import List, Literal, Dict, Any
from fastapi import UploadFile, HTTPException

from backend.app.api.db import get_transcript_by_id, save_transcript, update_transcript_fields
from backend.app.core.langgraph_flow import TranscriptWorkflow
from backend.app.transcript.transcription import get_transcript
from backend.app.utils.config import get_config
from backend.app.agents.chatbot import Chatbot
from backend.app.api.schemas import ChatResponse, ConversationHistoryResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TaskType = Literal["summarize", "highlight", "violation"]

async def process_youtube(
        url: str, 
        captions: bool, 
        provider: str, 
        model: str, 
        language: str
) -> dict:
    """
    Process transcript for YouTube video.
    """
    config = get_config({
        "type_of_source": "YouTube Video",
        "source_url_or_path": url,
        "use_youtube_captions": captions,
        "transcription_method": "Cloud Whisper",
        "language": language,
        "provider": provider,
        "model": model
    })
    transcript = get_transcript(config)
    transcript_id = await save_transcript(transcript, config)
    return {
        "id": transcript_id.id,
        "transcript": transcript,
        "config": config
    }

async def process_upload(
        file: UploadFile, 
        language: str
    ) -> dict:
    """
    Process transcript for uploaded file.
    """
    if not file.filename.lower().endswith(('.mp3', '.mp4', '.wav', '.m4a', '.webm')):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_file_path = tmp.name
        tmp.write(await file.read())

    try:
        config = get_config({
            "type_of_source": "Local File",
            "source_url_or_path": temp_file_path,
            "language": language,
        })
        transcript = get_transcript(config)
        transcript_id = await save_transcript(transcript, config)
        return {
            "id": transcript_id.id,
            "transcript": transcript,
            "config": config
        }
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

async def agents(
        transcript_id: str,
        task_type: List[TaskType]
) -> Dict[str, Any]:
    """
    Run specified agents on the transcript.
    """
    transcript_record = await get_transcript_by_id(transcript_id)
    if not transcript_record:
        raise HTTPException(status_code=404, detail="Transcript not found")

    workflow = TranscriptWorkflow()
    transcript_text = transcript_record.transcript
    field_map = {
        "summarize": "summary",
        "highlight": "highlights",
        "violation": "violations",
    }

    results: Dict[str, Any] = {}
    for task_type in task_type:
        if task_type not in field_map:
            raise HTTPException(status_code=400, detail=f"Invalid task type: {task_type}")

        out = await workflow.run(transcript_text=transcript_text, task_type=task_type)
        field_name = field_map[task_type]
        results[field_name] = out.get(field_name)

        await update_transcript_fields(transcript_id, **{field_name: results[field_name]})

    return {
        "id": transcript_id,
        "tasks": task_type,
        "results": results
    }

chat_instances: Dict[str, Chatbot] = {}

def get_or_create_chatbot(session_id: str) -> Chatbot:
    """Get existing chatbot instance or create new one"""
    if session_id not in chat_instances:
        chatbot = Chatbot()
        chatbot.set_session_id(session_id)
        chat_instances[session_id] = chatbot
    return chat_instances[session_id]

async def add_transcript_to_vectorstore(transcript_id: str) -> bool:
    """Add transcript to vector store"""
    try:
        transcript_record = await get_transcript_by_id(transcript_id)
        if not transcript_record:
            logger.error(f"Transcript not found: {transcript_id}")
            return False
        
        # Use any chatbot instance to access vector store
        chatbot = Chatbot()
        success = chatbot.add_transcript_to_vectorstore(transcript_id, transcript_record.transcript)
        return success
    except Exception as e:
        logger.error(f"Error adding transcript to vector store: {e}")
        return False

async def chat_ask(session_id: str, question: str, include_history: bool = False) -> ChatResponse:
    """Ask chatbot a question"""
    try:
        chatbot = get_or_create_chatbot(session_id)
        response = chatbot.ask(question, include_history)
        
        return ChatResponse(
            session_id=session_id,
            question=question,
            response=response,
            response_type="standard"
        )
    except Exception as e:
        logger.error(f"Error in chat_ask: {e}")
        raise HTTPException(status_code=500, detail="Error processing chat request")

async def chat_ask_with_rag(session_id: str, question: str, include_history: bool = False) -> ChatResponse:
    """Ask chatbot with RAG"""
    try:
        chatbot = get_or_create_chatbot(session_id)
        response = chatbot.ask_with_rag(question, include_history)
        
        return ChatResponse(
            session_id=session_id,
            question=question,
            response=response,
            response_type="rag"
        )
    except Exception as e:
        logger.error(f"Error in chat_ask_with_rag: {e}")
        raise HTTPException(status_code=500, detail="Error processing RAG chat request")

async def clear_chat_history(session_id: str) -> bool:
    """Clear chat conversation history"""
    try:
        if session_id in chat_instances:
            chat_instances[session_id].clear_conversation_history()
            return True
        return False
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return False