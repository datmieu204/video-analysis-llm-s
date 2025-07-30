# backend/app/transcript/sources/audio_base.py

import os
import tempfile
from typing import Optional

class AudioSourceHandler:
    """
    Abstract base class for audio source handlers.
    """
    def __init__(self, source_path: str, temp_dir: Optional[str] = None):
        self.source_path = source_path
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
    def get_processed_audio(self) -> str:
        """
        Returns the processed audio path.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def clean_up(self, file_path: str) -> None:
        """
        Cleans up a temporary file if it exists.
        """
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")
