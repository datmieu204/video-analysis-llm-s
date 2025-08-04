# backend/app/core/config.py

import itertools

from langchain_google_genai import ChatGoogleGenerativeAI
from backend.app.utils.config import GOOGLE_API_KEYS

class MultiLLMs:
    def __init__(self, api_keys=GOOGLE_API_KEYS, model="gemini-1.5-flash", **kwargs):
        self.api_keys = api_keys
        self.model = model
        self.kwargs = kwargs
        self.key_selector = itertools.cycle(api_keys)

    def get_llm(self):
        api_key = next(self.key_selector)
        return ChatGoogleGenerativeAI(
            model=self.model,
            api_key=api_key,
            **self.kwargs
        )
    
    def invoke(self, prompt: str):
        llm = self.get_llm()
        response = llm.invoke(prompt)
        return response
    
    async def ainvoke(self, prompt: str):
        llm = self.get_llm()
        response = await llm.ainvoke(prompt)
        return response