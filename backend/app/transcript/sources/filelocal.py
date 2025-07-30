# backend/app/transcript/sources/filelocal.py

import os
from backend.app.transcript.sources.audio_base import AudioSourceHandler
from backend.app.utils.ffmpeg_utils import convert_to_wav, process_audio_file

class LocalAudioHandler(AudioSourceHandler):
    """
    Handles local audio files.
    """
    def get_processed_audio(self) -> str:
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Cannot find local file: {self.source_path}")
        
        if not self.source_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.mp3', '.wav')):
            raise ValueError("Unsupported file format. Supported formats: mp4, mkv, avi, mov, mp3, wav")

        temp_wav = os.path.join(self.temp_dir, "local_audio.wav")
        convert_to_wav(self.source_path, temp_wav)
        
        processed_path = os.path.join(self.temp_dir, "local_processed.mp3")
        process_audio_file(temp_wav, processed_path)
        
        self.clean_up(temp_wav)

        return processed_path