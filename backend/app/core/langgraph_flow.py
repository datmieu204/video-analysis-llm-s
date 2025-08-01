# backend/app/core/langgraph_flow.py

from typing import Dict, Any, List, Tuple
from backend.app.agents.summarizer import Summarizer

async def summarize_node(state: dict) -> dict:
    summarizer = Summarizer(api_keys=state["api_keys"])
    text = state["transcript_text"]
    summary = await summarizer.summarize_chunks(text)
    state["summary_text"] = summary
    return state