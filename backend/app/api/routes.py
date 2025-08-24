# backend/app/api/routes.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
from backend.app.api.db import get_transcript_by_id
from backend.app.api.services import process_youtube, process_upload, agents, TaskType
from backend.app.api.schemas import TranscriptResponse
from backend.app.core.langgraph_flow import TranscriptWorkflow
from backend.app.api.schemas import (
    ChatRequest, 
    ChatRagRequest,
    ChatResponse,
    AddTranscriptRequest,
    ConversationHistoryResponse
)
from backend.app.api.services import (
    chat_ask,
    chat_ask_with_rag,
    clear_chat_history,
    add_transcript_to_vectorstore,
)

router = APIRouter()

@router.post("/transcript/youtube")
async def transcript_youtube(
    url: str = Form(..., description="YouTube video URL"),
    captions: bool = Form(True, description="Use YouTube captions if available"),
    provider: str = Form("GROQ", description="Provider for the transcription service"),
    model: str = Form("llama-3.3-70b-versatile", description="Model to use for transcription"),
    language: str = Form("en", description="Language code for transcription")
):
    try:
        return await process_youtube(url, captions, provider, model, language)
    except HTTPException as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/transcript/upload")
async def transcript_upload(
    file: UploadFile = File(..., description="Audio or video file to transcribe"),
    language: str = Form("en", description="Language code for transcription")
):
    try:
        return await process_upload(file, language)
    except HTTPException as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/transcript/{transcript_id}")
async def get_transcript_(transcript_id: str):
    try:
        transcript_record = await get_transcript_by_id(transcript_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid transcript ID")
    
    if not transcript_record:
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    return {
        "id": transcript_record.id,
        "transcript": transcript_record.transcript,
        "config": transcript_record.config
    }

@router.post("/transcript/agents/{transcript_id}")
async def run_agents(transcript_id: str, task_type: List[TaskType] = Form(...)):
    return await agents(transcript_id, task_type)

@router.post("/chat/add-transcript", response_model=dict)
async def add_transcript_endpoint(request: AddTranscriptRequest):
    """Add transcript to vector store"""
    try:
        result = await add_transcript_to_vectorstore(request.transcript_id)
        return {"success": result, "message": "Transcript added to vector store" if result else "Failed to add transcript"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/ask", response_model=ChatResponse)
async def chat_ask_endpoint(request: ChatRequest):
    """Ask chatbot a question"""
    try:
        response = await chat_ask(request.session_id, request.question, request.include_history)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/ask-rag", response_model=ChatResponse)
async def chat_ask_rag_endpoint(request: ChatRagRequest):
    """Ask chatbot with RAG"""
    try:
        response = await chat_ask_with_rag(request.session_id, request.question, request.include_history)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/chat/clear/{session_id}")
async def clear_chat_endpoint(session_id: str):
    """Clear chat conversation history"""
    try:
        success = await clear_chat_history(session_id)
        return {"success": success, "message": "Chat history cleared" if success else "Failed to clear chat history"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
