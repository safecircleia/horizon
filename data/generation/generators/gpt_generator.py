"""OpenAI GPT-based conversation generator."""

import os
from typing import Optional
import openai
from data.generation.generators.base import ConversationGenerator, GenerationResult
from data.generation.prompts.base import ConversationPrompt


class GPTGenerator(ConversationGenerator):
    """Generate conversations using OpenAI GPT API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.9,
        max_tokens: int = 2000
    ):
        """Initialize GPT generator.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: GPT model to use
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = openai.AsyncOpenAI(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "openai"

    async def generate(self, prompt: ConversationPrompt) -> GenerationResult:
        """Generate conversation using OpenAI API."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": prompt.system_prompt},
                    {"role": "user", "content": prompt.user_prompt}
                ],
                response_format={"type": "json_object"}  # Force JSON output
            )

            # Extract text from response
            response_text = response.choices[0].message.content

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
