# backend/app/agents/highlighter.py

import asyncio
import json
from typing import List, Dict, Any
from backend.app.core.multi_llms import MultiLLMs
from backend.app.utils.nlp_utils import prompt_highlights, score_selection
from backend.app.prompts.highlight_prompt import highlight_prompt
import logging

logger = logging.getLogger(__name__)

semaphore = asyncio.Semaphore(6)

class Highlighter:
    def __init__(self, **kwargs):
        self.multi_llms = MultiLLMs(**kwargs)

    async def highlight_text(self, text: str) -> List[Dict[str, Any]]:
        async with semaphore:
            try:
                candidate_highlights = score_selection(text, top_n=5)
                
                if not candidate_highlights:
                    logging.info("No candidate highlights found after scoring.")
                    return []

                prompt_text = prompt_highlights(candidate_highlights)
                
                if not prompt_text:
                    logging.info("Generated prompt text is empty.")
                    return []
                
                prompt = highlight_prompt.format_prompt(text=prompt_text).to_string()
                response = await self.multi_llms.ainvoke(prompt)

                response_text = getattr(response, 'content', response)
                if response_text is None:
                    logging.error("LLM response content is None")
                    return []

                cleaned_response = response_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

                try:
                    highlights_json = json.loads(cleaned_response)
                    return highlights_json
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to decode JSON from LLM response: {e}\nResponse: {cleaned_response}")
                    return []
            
            except Exception as e:
                logging.error(f"An unexpected error occurred in highlight_text: {e}", exc_info=True)
                return []