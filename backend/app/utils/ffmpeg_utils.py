# backend/app/utils/ffmpeg_utils.py
# -*- coding: utf-8 -*-

import subprocess
from typing import List, Optional

def process_audio_file(input_path: str, output_path: str) -> None:
    """
    Convert audio to low-quality MP3 for API limits.
    """
    command = [
        'ffmpeg', '-y', '-i', input_path,
        '-ar', '16000',
        '-ac', '1',
        '-b:a', '32k',
        output_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing audio file: {e.stderr}")

def convert_to_wav(input_path: str, output_wav: str, ffmpeg_args: Optional[List[str]] = None) -> None:
    """
    Extract and convert audio to WAV format.
    """
    args = ['ffmpeg', '-y', '-i', input_path, '-vn']

    if ffmpeg_args:
        args.extend(ffmpeg_args)
    else:
        args.extend(['-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1'])
    args.append(output_wav)

    try:
        subprocess.run(args, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting to WAV: {e.stderr}")