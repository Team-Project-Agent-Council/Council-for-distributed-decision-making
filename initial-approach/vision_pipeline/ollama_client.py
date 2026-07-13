"""Factory for a shared Ollama client instance."""

from ollama import Client

from vision_pipeline.config import PipelineConfig


def make_ollama_client(config: PipelineConfig) -> Client:
    """Create an Ollama client connected to the configured host."""
    return Client(host=config.ollama_host)
