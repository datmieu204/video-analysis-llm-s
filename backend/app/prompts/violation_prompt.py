# backend/app/prompts/violation_prompt.py

from langchain_core.prompts import PromptTemplate

violation_prompt = PromptTemplate(
    input_variables=["relevant_law_docs", "text"],
    template="""
    You are a legal expert analyzing potential violations in a transcript.
    Based on the following relevant legal documents:
    ---
    {relevant_law_docs}
    ---

    Analyze the following transcript. The text contains timestamps in the [HH:MM:SS] format.
    ---
    {text}
    ---

    Your task is to:
    1. Identify any behavior described in the text that shows signs of a legal violation.
    2. For each violation found, extract the nearest timestamp [HH:MM:SS] to the sentence describing the behavior.
    3. Provide a brief explanation of why it is a violation, based on the provided legal documents.

    Return the result as a JSON list. Each object in the list must have the following fields:
    - "timestamp": The timestamp [HH:MM:SS] of the violation.
    - "violation": A brief description of the violating behavior.
    - "explanation": A brief explanation of why it is a violation, citing the law if possible.

    Example response format:
    [
        {{
            "timestamp": "[00:05:32]",
            "violation": "Threatening to use violence.",
            "explanation": "This behavior shows signs of 'Criminal Threat' as the speaker expresses intent to cause serious harm to another person."
        }}
    ]

    If no violations are found, return an empty list: [].
    """
)
