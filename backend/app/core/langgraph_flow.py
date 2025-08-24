# backend/app/core/langgraph_flow.py

import logging

from langgraph.graph import StateGraph, END
from backend.app.api.schemas import WorkflowState, HighlightItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptWorkflow:
    def __init__(self):
        """
        Init graph for transcript workflow.
        """
        self.graph = StateGraph(dict)

        self.graph.add_node("router", self.router_node)
        self.graph.add_node("summarizer", self.summarize_node)
        self.graph.add_node("highlighter", self.highlight_node)
        self.graph.add_node("violence_detector", self.violation_node)

        self.graph.set_entry_point("router")

        self.graph.add_conditional_edges(
            "router",
            lambda state: state["next"],
            {
                "summarizer": "summarizer",
                "highlighter": "highlighter",
                "violence_detector": "violence_detector"
            }
        )

        self.graph.add_edge("summarizer", END)
        self.graph.add_edge("highlighter", END)
        self.graph.add_edge("violence_detector", END)

        self.compiled = self.graph.compile()

    async def summarize_node(self, state: dict) -> dict:
        from backend.app.agents.summarizer import Summarizer
        summarizer = Summarizer()
        text = state["transcript_text"]
        summary = await summarizer.summarize_chunks(text)
        return {"summary": summary.content}

    async def highlight_node(self, state: dict) -> dict:
        from backend.app.agents.highlighter import Highlighter
        highlighter = Highlighter()
        text = state["transcript_text"]
        highlights = await highlighter.highlight_text(text)
        highlights_clean = [
            highlight if isinstance(highlight, dict) else highlight.model_dump()
            for highlight in highlights
        ]
        return {"highlights": highlights_clean}

    async def violation_node(self, state: dict) -> dict:
        from backend.app.agents.violence_detecter import ViolenceDetector
        violence_detector = ViolenceDetector()
        text = state["transcript_text"]
        violations = await violence_detector.analyze_transcript_chunks(text)
        return {"violations": violations}

    def router_node(self, state: dict) -> dict:
        """
        Route the workflow to the appropriate node based on the task type.
        """
        task_type = state.get("task_type", "summarize")
        next_node = "summarizer"
        if task_type == "highlight":
            next_node = "highlighter"
        elif task_type == "violation":
            next_node = "violence_detector"

        state["next"] = next_node
        return state

    async def run(self, transcript_text: str, task_type: str) -> dict:
        state = {
            "transcript_text": transcript_text,
            "task_type": task_type
        }
        return await self.compiled.ainvoke(state)
