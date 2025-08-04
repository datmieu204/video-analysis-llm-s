# backend/app/agents/summarizer.py

import asyncio

from typing import List
from backend.app.prompts.summarize_prompt import summarize_chunk_prompt, summarize_merge_prompt
from backend.app.core.multi_llms import MultiLLMs
from backend.app.utils.splitter import text_splitter

class Summarizer:
    def __init__(self, api_keys: List[str], model: str = "gemini-1.5-flash", chunk_size: int = 10000, **kwargs):
        self.multi_llms = MultiLLMs(api_keys, model=model, **kwargs)
        self.chunk_size = chunk_size

    async def summarize_chunk(self, text: str) -> str:
        """
        Asynchronously summarizes a single chunk of text.
        
        Args:
            text (str): The text to be summarized.
        
        Returns:
            str: The summary of the text chunk.
        """
        prompt = summarize_chunk_prompt.format_prompt(text=text)
        return await self.multi_llms.ainvoke(prompt)

    async def summarize_chunks(self, text: str) -> str:
        """
        Asynchronously summarizes a long text by splitting it into chunks and summarizing each chunk.
        
        Args:
            text (str): The long text to be summarized.
        
        Returns:
            str: The final summary of the text.
        """
        chunks = text_splitter(text, chunk_size=self.chunk_size)

        chunk_summaries = await asyncio.gather(
            *[self.summarize_chunk(chunk) for chunk in chunks]
        )
        
        chunk_summaries_texts = [msg.content for msg in chunk_summaries]

        prompt = summarize_merge_prompt.format_prompt(chunk_summaries="\n".join(chunk_summaries_texts)).to_string()

        return await self.multi_llms.ainvoke(prompt)