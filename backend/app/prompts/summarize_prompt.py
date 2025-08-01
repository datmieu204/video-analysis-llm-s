# backend/app/prompts/summarize_chunk_prompt.py

from langchain.prompts import PromptTemplate

summarize_chunk_prompt = PromptTemplate.from_template("""
You are given a section of a video transcript. Summarize it concisely and clearly.

Transcript:
{text}

Summary:
""")

summarize_merge_prompt = PromptTemplate.from_template("""
Below are several chunk-level summaries of a long video. Merge them into a cohesive summary with natural flow and structure.

Chunk summaries:
{chunk_summaries}

Final summary:
""")
