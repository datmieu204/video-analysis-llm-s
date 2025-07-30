# backend/app/transcript/transcription.py

import os
from backend.app.transcript.whisper import transcribe_audio
from backend.app.transcript.sources.audio_base import AudioSourceHandler
from backend.app.transcript.sources.filelocal import LocalAudioHandler
from backend.app.transcript.sources.youtube import YouTubeAudioHandler
from backend.app.transcript.sources.youtube import get_youtube_id, get_youtube_transcript


def get_source_type(source_type: str, source_path: str) -> AudioSourceHandler:
    """
    Get the audio source type based on the source path.
    """
    types = {
        "YouTube Video": YouTubeAudioHandler,
        "Local File": LocalAudioHandler,
    }
    
    try:
        type_class = types.get(source_type)
        if not type_class:
            raise ValueError(f"Not supported: {source_type}")
    except KeyError:
        raise ValueError(f"Invalid source type: {source_type}")
        
    return type_class(source_path)


def get_transcript(config: dict) -> str:
    """
    Get transcript from video source.
    """
    source_type = config.get("type_of_source")
    source_path = config.get("source_url_or_path")
    
    if not source_type or not source_path:
        raise ValueError("Type of source and source URL/path are required")
    
    if source_type == "YouTube Video":
        try:
            if config.get("use_youtube_captions", True):
                video_id = get_youtube_id(source_path)
                return get_youtube_transcript(video_id, config.get("language")) # "en"
        except Exception as e:
            config["use_youtube_captions"] = False

        audio_path = get_source_type(source_type, source_path).get_processed_audio()

        try:
            return transcribe_audio(audio_path, config)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
    else:
        try:
            source_audio = get_source_type(source_type, source_path)
            audio_path = source_audio.get_processed_audio()

            try:
                return transcribe_audio(audio_path, config)
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
        except Exception as e:
            raise Exception(f"Error processing audio source: {e}")

