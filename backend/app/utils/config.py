# backend/app/utils/config.py

import os
import re
import itertools

from urllib.parse import urlparse, parse_qs
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR_TRANSCRIPTS = "backend/app/saved_transcripts"

DEFAULT_CONFIG = {
    "type_of_source": "YouTube Video",
    "youtube_captions": True,
    "transcription_method": "Cloud Whisper",
    "language": "auto",
    "chunk_size": 5000,
    "parallel_api_calls": 10,
    "max_output_tokens": 4096,
    "provider": "GROQ",
    "base_url": "https://api.groq.com/openai/v1",
    "model": "llama-3.3-70b-versatile"
}

GOOGLE_API_KEYS = [key for key in (
    os.environ.get("GOOGLEAI_API_KEY"),
    os.environ.get("GOOGLEAI_API_KEY_1"),
    os.environ.get("GOOGLEAI_API_KEY_2"),
    os.environ.get("GOOGLEAI_API_KEY_3"),
    os.environ.get("GOOGLEAI_API_KEY_4"),
) if key]

GROQ_API_KEYS = [key for key in (
    os.environ.get("GROQ_API_KEY"),
    os.environ.get("GROQ_API_KEY_1"),
    os.environ.get("GROQ_API_KEY_2"),
) if key]

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class APIKeys:
    def __init__(self, keys: list = GROQ_API_KEYS):
        self.keys = keys
        self.key_selector = itertools.cycle(keys)

    def get_key(self) -> str:
        if not self.keys:
            raise ValueError("No API keys available.")
        return next(self.key_selector)

def get_api_key(config: Dict, name_api_key: str) -> str:
    """
    Get the API key from environment variables or configuration.
    """
    if not name_api_key:
        name_api_key = "GROQ_API_KEY"
    
    keys = []

    for key, value in os.environ.items():
        if key.startswith(name_api_key) and value.strip():
            keys.append(value)

    if not keys and "api_key" in config:
        raise ValueError(f"API key not found in environment variables or configuration for {name_api_key}.")

    api_key = APIKeys(keys=keys).get_key()

    if not api_key:
        raise ValueError("Not found API key. Please set the environment variable or provide it in the configuration.")
    print(f"Using API key from {name_api_key} environment variable or configuration.")
    return api_key

def get_config(user_config: Dict) -> Dict:
    """
    Combine user-provided configuration with default settings.
    """
    config = DEFAULT_CONFIG.copy()
    config.update(user_config)
    print("Merged configuration:", config)
    return config

def get_output_path(config: Dict) -> str:
    """
    Generate the output path for the transcript file based on the source URL or path.
    """
    source = config.get("source_url_or_path", "")
    source_type = config.get("type_of_source", "").lower()

    def sanitize(name: str) -> str:
        return re.sub(r'[<>:"/\\|?*]', '_', name)

    if "youtube" in source_type and "youtube.com" in source:
        query = urlparse(source).query
        video_id = parse_qs(query).get("v", ["youtube"])[0]
        return f"transcript_youtube_{sanitize(video_id)}.txt"
    elif "dropbox" in source_type and "dropbox.com" in source:
        filename = Path(urlparse(source).path).name
        return f"transcript_dropbox_{sanitize(filename)}.txt"
    elif "local" in source_type or os.path.exists(source):
        filename = Path(source).stem
        return f"transcript_local_{sanitize(filename)}.txt"
    else:
        fallback = sanitize(Path(source).stem or "unknown")
        return f"transcript_{fallback}.txt"
