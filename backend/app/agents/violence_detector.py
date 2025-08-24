# backend/app/agents/violence_detecter.py

import asyncio
import json
import logging

from typing import List, Dict, Any
from backend.app.core.multi_llms import MultiLLMs
from backend.app.core.embeddings import VectorStore
from backend.app.prompts.violation_prompt import violation_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from backend.app.utils.splitter import text_splitter

class ViolenceDetector:
    def __init__(self, **kwargs):
        self.multi_llms = MultiLLMs(**kwargs)
        self.vector_store = VectorStore(**kwargs)
        self.vector_store.load_vectorstore()
        self.semaphore = asyncio.Semaphore(8)

    async def detect_violence(self, text: str) -> List[Dict[str, Any]]:
        """
        Detects violent content in the provided text.
        """

        if not self.vector_store.vectorstore:
            print("Vector store is not loaded. Please ensure the vector store is built and persisted.")
            return []

        try:
            query_embeddings = self.vector_store.query_embeddings.embed_query(text)

            relevant_law_docs = self.vector_store.vectorstore.similarity_search_by_vector(
                embedding=query_embeddings,
                k=3
            )  

            relevant_law_docs_text = "\n\n".join([doc.page_content for doc in relevant_law_docs])

            prompt = violation_prompt.format(
                relevant_law_docs=relevant_law_docs_text,
                text=text
            )

            response = await self.multi_llms.ainvoke(prompt)

            return self.parse_response(response.content)

        except Exception as e:
            print(f"Error in detect_violence: {e}")
            return []
        
    def parse_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parses the LLM response (expected in JSON format) to extract violations.
        """
        if isinstance(response, tuple):
            response = response[1]

        cleaned_response = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            violations = json.loads(cleaned_response)
            if isinstance(violations, list):
                return violations
            else:
                print("Unexpected response format: Expected a list of violations.")
                return []
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from LLM response: {e}\nResponse: {cleaned_response}")
            return []
        
    async def analyze_transcript_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Runs the violence detection process.
        """
        chunks = text_splitter(text)

        async def analyze_with_semaphore(chunk: str) -> List[Dict[str, Any]]:
            async with self.semaphore:
                result = await self.detect_violence(chunk)
                await asyncio.sleep(1) 
                return result

        tasks = [analyze_with_semaphore(chunk) for chunk in chunks]
        result = await asyncio.gather(*tasks)
        return [item for sublist in result for item in sublist]