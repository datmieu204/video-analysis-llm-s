# backend/app/transcript/sources/whisper.py

from typing import Dict
from backend.app.utils.formatting import format_time
from backend.app.utils.config import DEFAULT_CONFIG, get_api_key

def transcribe_audio(audio_path: str, config: Dict) -> str:
    """
    Transcribe audio file using Whisper (Cloud or Local).
    """
    method = config.get("transcription_method", "Cloud Whisper")
    language = config.get("language", None)

    try:
        if (method == "Cloud Whisper"):
            return _transcribe_with_groq(audio_path, language)
        elif (method == "Local Whisper"):
            return _transcribe_with_local_whisper(audio_path, language)
        else:
            raise ValueError(f"Unknown transcription method: {method}")
    except Exception as e:
        raise Exception(f"Transcription error: {e}")


def _transcribe_with_groq(audio_path: str, language: str = "en") -> str:
    """
    Using Groq API for transcription.
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Please install 'groq' to use Cloud Whisper.")

    api_key = get_api_key(DEFAULT_CONFIG, "GROQ_API_KEY")

    groq_client = Groq(api_key=api_key)
    
    with open(audio_path, "rb") as audio_file:
        print("\nStarting transcription with Groq API...")

        response = groq_client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="verbose_json",
            language=language if language else "en",
            temperature=0.0
        )

        transcript = ""
        if hasattr(response, "segments"): # Check if response has segments attribute
            for segment in response.segments:
                timestamp = format_time(segment['start'])
                transcript += f"{timestamp} {segment['text'].strip()}\n"
        else:
            raise ValueError("Unexpected response format from Groq API")

        return transcript

def _transcribe_with_local_whisper(audio_path: str, language: str = None) -> str:
    """
    Using local Whisper library for transcription.
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("Please install 'openai-whisper' to use Local Whisper.")

    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("Starting transcription with Local Whisper...")
    result = model.transcribe(audio_path, language=language)

    transcript = ""

    for segment in result["segments"]:
        time = format_time(segment["start"])
        transcript += f"{time} {segment['text'].strip()}\n"

    return transcript