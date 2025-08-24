# backend/app/api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class WorkflowState(BaseModel):
    transcript_text: str
    task_type: str

class HighlightItem(BaseModel):
    timestamp: str
    text: str
    reason: str

class ViolenceItem(BaseModel):
    timestamp: Optional[str]
    violation: str
    explanation: str

class TranscriptConfig(BaseModel):
    type_of_source: str = Field(..., description="Type of source for the transcript (e.g., 'video', 'audio')")
    source_url_or_path: str = Field(..., description="URL or path to the source")
    use_youtube_captions: bool = Field(True, description="Whether to use YouTube captions if available")
    transcription_method: str = Field("Cloud Whisper", description="Method used for transcription (e.g., 'Cloud Whisper', 'Local Whisper')")
    language: str = Field("en", description="Language code for transcription (e.g., 'en', 'es', 'fr')")
    provider: str = Field("GROQ", description="Provider for the transcription service (e.g., 'GROQ', 'Google')")
    parallel_api_calls: int = Field(10, description="Number of parallel API calls to make")
    max_output_tokens: int = Field(4096, description="Maximum number of output tokens for the transcription")
    base_url: str = Field(..., description="Base URL for the transcription service API")
    model : str = Field("llama-3.3-70b-versatile", description="Model to use for transcription")

class TranscriptResponse(BaseModel):
    id: Optional[str] = Field(None, description="Unique identifier for the transcript")
    title: Optional[str] = Field(None, description="Title of the transcript")
    transcript: str = Field(..., description="The generated transcript text")
    config: TranscriptConfig = Field(..., description="Configuration used for the transcription")
    summary: Optional[str] = Field(None, description="Summary of the transcript")
    highlights: Optional[List[HighlightItem]] = Field(None, description="List of highlights from the transcript")
    violations: Optional[List[ViolenceItem]] = Field(None, description="List of violations detected in the transcript")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the transcript was created")
    updated_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the transcript was last updated")

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Chat session ID")
    question: str = Field(..., description="User question")
    include_history: bool = Field(False, description="Include conversation history in context")

class ChatRagRequest(BaseModel):
    session_id: str = Field(..., description="Chat session ID")
    question: str = Field(..., description="User question")
    include_history: bool = Field(False, description="Include conversation history in context")

class ChatResponse(BaseModel):
    session_id: str = Field(..., description="Chat session ID")
    question: str = Field(..., description="User question")
    response: str = Field(..., description="Chatbot response")
    response_type: str = Field(..., description="Type of response (standard, rag)")

class AddTranscriptRequest(BaseModel):
    transcript_id: str = Field(..., description="Transcript ID to add to vector store")

class ConversationHistoryResponse(BaseModel):
    session_id: str = Field(..., description="Chat session ID")
    history: List[Dict[str, Any]] = Field(..., description="Conversation history")