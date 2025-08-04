# backend/app/prompts/highlight_prompt.py

from langchain.prompts import PromptTemplate

highlight_prompt = PromptTemplate.from_template("""
You are an expert in analyzing video transcripts to find meaningful highlights.

You are given a transcript containing multiple groups. Each group has timestamps, sentences, and extracted keywords.

### What is a highlight?
A highlight is a moment that stands out to viewers — it is interesting, emotionally engaging, funny, insightful, surprising, or contains important information. 
It could be a key moment that summarizes an idea, a personal story, a conflict, a joke, or a memorable statement.
There may be multiple highlights in the same group, or none at all.

### Your task:
1. Carefully analyze the entire transcript.
2. Identify **up to 3 highlights** — the most impactful and attention-grabbing moments in this transcript.
3. For each highlight, select the **single most important sentence** (with its timestamp) that represents the highlight.
4. Briefly explain **why** this sentence is a highlight.

### Output format:
Return a **valid JSON list** in this exact format:

[
  {{
    "timestamp": "<timestamp of the most representative sentence>",
    "text": "<the most representative sentence>",
    "reason": "<briefly explain why this is a highlight>"
  }}
]

### Notes:
- If there are no highlights, return an empty list: []
- If you find fewer than 3 highlights, return only those.
- Be concise and focus on what would genuinely interest viewers.

### Transcript:
{text}
    """
)