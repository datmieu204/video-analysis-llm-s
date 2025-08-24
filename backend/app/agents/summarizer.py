# backend/app/agents/summarizer.py

import asyncio
import logging

from typing import List
from backend.app.prompts.summarize_prompt import summarize_chunk_prompt, summarize_merge_prompt
from backend.app.core.multi_llms import MultiLLMs
from backend.app.utils.splitter import text_splitter
from backend.app.utils.config import GOOGLE_API_KEYS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(
        self, 
        api_keys: List[str] = GOOGLE_API_KEYS, 
        model: str = "gemini-1.5-flash", 
        chunk_size: int = 10000, **kwargs
    ):
        self.multi_llms = MultiLLMs(api_keys, model=model, **kwargs)
        self.chunk_size = chunk_size

    async def summarize_chunk(self, text: str) -> str:
        """
        Asynchronously summarizes a single chunk of text.
        """
        try:
            prompt = summarize_chunk_prompt.format_prompt(text=text)
            return await self.multi_llms.ainvoke(prompt)
        except Exception as e:
            logger.error(f"Error summarizing chunk: {e}")
            return f"Error summarizing chunk: {e}"

    async def summarize_chunks(self, text: str) -> str:
        """
        Asynchronously summarizes a long text by splitting it into chunks and summarizing each chunk.
        """
        chunks = text_splitter(text, chunk_size=self.chunk_size)
        if not chunks:
            logger.warning("No chunks to summarize.")
            return "No content to summarize."
        logger.info(f"Summarizing {len(chunks)} chunks of text.")

        chunk_summaries = await asyncio.gather(
            *[self.summarize_chunk(chunk) for chunk in chunks],
            return_exceptions=True
        )

        chunk_summaries_texts = [msg.content for msg in chunk_summaries if not isinstance(msg, Exception)]

        prompt = summarize_merge_prompt.format_prompt(chunk_summaries="\n".join(chunk_summaries_texts)).to_string()

        return await self.multi_llms.ainvoke(prompt)