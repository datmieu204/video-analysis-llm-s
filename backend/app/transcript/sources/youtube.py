# backend/app/transcript/sources/youtube.py

import os
import re
import tempfile
import pytubefix as pytube
from youtube_transcript_api import YouTubeTranscriptApi
from backend.app.utils.formatting import format_time
from backend.app.utils.ffmpeg_utils import process_audio_file
from backend.app.transcript.sources.audio_base import AudioSourceHandler

def get_youtube_id(url: str) -> str:
    """
    Extract YouTube video ID from a URL.
    """
    regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(regex, url)

    if not match:
        raise ValueError("Cannot extract YouTube video ID")
    
    return match.group(1)

def get_youtube_transcript(video_id: str, language: str = "en") -> str:
    """
    Get transcript from YouTube subtitles.
    """
    try:
        if language == "auto":
            language = "en"

        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=[language]
        )

        return "\n".join(f"{format_time(entry['start'])} {entry['text'].strip()}" for entry in transcript_list)
    except Exception as e:
        raise Exception(f"Cannot get YouTube transcript. Error: {e}")

def download_youtube_audio(url: str) -> str:
    """
    Download audio from YouTube video and process it.
    """
    try:
        yt = pytube.YouTube(url)
        stream = yt.streams.get_audio_only()
        
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "yt_audio.mp4")
        processed_path = os.path.join(temp_dir, "yt_audio_processed.mp3")
        
        stream.download(output_path=temp_dir, filename="yt_audio.mp4")
        process_audio_file(temp_path, processed_path)
        os.remove(temp_path)
        
        return processed_path
    except Exception as e:
        raise Exception(f"Cannot download YouTube audio: {e}")      
    
class YouTubeAudioHandler(AudioSourceHandler):
    """
    Handles YouTube audio sources.
    """
    def get_processed_audio(self) -> str:
        """
        Returns the processed audio path.
        """
        if not self.source_path:
            raise ValueError("YouTube source path is required")
        
        return download_youtube_audio(self.source_path)