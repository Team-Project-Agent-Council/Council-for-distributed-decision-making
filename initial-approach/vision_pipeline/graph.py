"""LangGraph pipeline for the vision detail extraction.

Pipeline:
    Scene Parser -> Detail Identifier -> Detail Extractor (DINO + Florence-2) -> Crop -> Focusser
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from vision_pipeline.config import PipelineConfig, load_config
from vision_pipeline.crop_tool import crop_tool
from vision_pipeline.detail_extractor import detail_extractor
from vision_pipeline.detail_focusser import detail_focusser
from vision_pipeline.detail_identifier import detail_identifier
from vision_pipeline.scene_parser import scene_parser
from vision_pipeline.state import PipelineState
from vision_pipeline.ollama_client import make_ollama_client


def _route_after_identifier(state: PipelineState) -> Literal["detail_extractor", "__end__"]:
    """Route after identifier: proceed to extraction if details found."""
    if state.get("has_details", False) and len(state.get("details", [])) > 0:
        return "detail_extractor"
    return END


def build_graph(config: PipelineConfig | None = None) -> StateGraph:
    """Build the vision pipeline.

    Scene Parser -> Detail Identifier -> Detail Extractor (DINO + Florence-2) -> Crop -> Focusser
    """
    if config is None:
        config = load_config()

    client = make_ollama_client(config)
    builder = StateGraph(PipelineState)

    builder.add_node("scene_parser", lambda s: scene_parser(s, client, config))
    builder.add_node("detail_identifier", lambda s: detail_identifier(s, client, config))
    builder.add_node("detail_extractor", lambda s: detail_extractor(s, config))
    builder.add_node("crop_tool", lambda s: crop_tool(s, config))
    builder.add_node("detail_focusser", lambda s: detail_focusser(s, client, config))

    builder.add_edge(START, "scene_parser")
    builder.add_edge("scene_parser", "detail_identifier")
    builder.add_conditional_edges(
        "detail_identifier",
        _route_after_identifier,
        {"detail_extractor": "detail_extractor", END: END},
    )
    builder.add_edge("detail_extractor", "crop_tool")
    builder.add_edge("crop_tool", "detail_focusser")
    builder.add_edge("detail_focusser", END)

    return builder.compile()
