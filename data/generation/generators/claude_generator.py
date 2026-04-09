"""Claude-based conversation generator."""

import os
from typing import Optional
import anthropic
from data.generation.generators.base import ConversationGenerator, GenerationResult
from data.generation.prompts.base import ConversationPrompt


class ClaudeGenerator(ConversationGenerator):
    """Generate conversations using Claude API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.9,
        max_tokens: int = 2000
    ):
        """Initialize Claude generator.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "claude"

    async def generate(self, prompt: ConversationPrompt) -> GenerationResult:
        """Generate conversation using Claude API."""
        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=prompt.system_prompt,
                messages=[
                    {"role": "user", "content": prompt.user_prompt}
                ]
            )

            # Extract text from response
            response_text = message.content[0].text

            # Parse JSON conversation
            conversation = self._parse_json_response(response_text)

            if conversation is None:
                return GenerationResult(
                    success=False,
                    conversation=None,
                    raw_response=response_text,
                    error="Failed to parse JSON from response"
                )

            # Validate required fields
            if "messages" not in conversation:
                return GenerationResult(
                    success=False,
                    conversation=None,
                    raw_response=response_text,
                    error="Response missing 'messages' field"
                )

            return GenerationResult(
                success=True,
                conversation=conversation,
                raw_response=response_text,
                error=None
            )

        except Exception as e:
            return GenerationResult(
                success=False,
                conversation=None,
                raw_response="",
                error=f"API Error: {str(e)}"
            )
