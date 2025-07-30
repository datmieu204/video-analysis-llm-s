# backend/app/utils/formatting.py

from typing import Dict, Any, List, Tuple

def format_time(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def format_time_youtube(timestamp: str, url: str) -> str:
    """
    Convert YouTube timestamp format to a link with timestamp.
    """
    try:
        hours, minutes, seconds = map(int, timestamp.split(':'))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return f"{timestamp} - {url}&t={total_seconds}s"
    except ValueError:
        return timestamp
    
# ----------------------------------------------------------

def parse_response_content(response: Dict[str, Any]) -> str:
    """
    Extract content from API response.
    """
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    return content

def format_summary_with_timestamps(summaries: List[Tuple[str, str]], config: Dict) -> str:
    """
    Format summaries with appropriate timestamp links.
    """
    formatted_pieces = []
    source_url = config.get("source_url_or_path", "")
    is_youtube = config.get("type_of_source") == "YouTube Video"
    
    for timestamp, summary in summaries:
        if timestamp:
            if is_youtube:
                try:
                    timestamp_text = format_time_youtube(timestamp, source_url)
                except:
                    timestamp_text = timestamp # Fallback
            else:
                timestamp_text = timestamp
            formatted_pieces.append(f"{timestamp_text}\n\n{summary}")
        else:
            formatted_pieces.append(summary)
            
    return "\n\n".join(formatted_pieces)

