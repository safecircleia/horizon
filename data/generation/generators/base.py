"""Base interface for conversation generators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from data.generation.prompts.base import ConversationPrompt


@dataclass
class GenerationResult:
    """Result from a conversation generation attempt."""
    success: bool
    conversation: Optional[Dict[str, Any]]
    raw_response: str
    error: Optional[str] = None


class ConversationGenerator(ABC):
    """Abstract base class for conversation generators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Generator name (e.g., 'claude', 'openai')."""
        pass

    @abstractmethod
    async def generate(self, prompt: ConversationPrompt) -> GenerationResult:
        """Generate a conversation from the given prompt.

        Args:
            prompt: ConversationPrompt with category, severity, and instructions

        Returns:
            GenerationResult with success status and conversation data
        """
        pass

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, handling common formatting issues.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        import json
        import re

        # Try direct JSON parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None
