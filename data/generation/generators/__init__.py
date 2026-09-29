"""LLM-based conversation generators."""

from data.generation.generators.base import ConversationGenerator, GenerationResult


def __getattr__(name):
    """Lazy-load optional generators to avoid import errors when dependencies are missing."""
    if name == "ClaudeGenerator":
        from data.generation.generators.claude_generator import ClaudeGenerator

        return ClaudeGenerator
    elif name == "GPTGenerator":
        from data.generation.generators.gpt_generator import GPTGenerator

        return GPTGenerator
    elif name == "BedrockGenerator":
        from data.generation.generators.bedrock_generator import BedrockGenerator

        return BedrockGenerator
    elif name == "VLLMGenerator":
        from data.generation.generators.vllm_generator import VLLMGenerator

        return VLLMGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConversationGenerator",
    "GenerationResult",
    "ClaudeGenerator",
    "GPTGenerator",
    "BedrockGenerator",
    "VLLMGenerator",
]
