"""Amazon Bedrock conversation generator using long-term API keys."""

import os
import json
from typing import Optional
from urllib.parse import quote
import httpx
from data.generation.generators.base import ConversationGenerator, GenerationResult
from data.generation.prompts.base import ConversationPrompt


class BedrockGenerator(ConversationGenerator):
    """Generate conversations using Amazon Bedrock Converse API with a long-term API key."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "eu.amazon.nova-micro-v1:0",
        region: str = "eu-west-3",
        temperature: float = 0.9,
        max_tokens: int = 2000,
    ):
        self.api_key = api_key or os.getenv("BEDROCK_API_KEY")
        if not self.api_key:
            raise ValueError("BEDROCK_API_KEY must be set")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Colons in model IDs must be percent-encoded in the URL path
        model_encoded = quote(model, safe="")
        self.endpoint = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_encoded}/converse"

    @property
    def name(self) -> str:
        return "bedrock"

    async def generate(self, prompt: ConversationPrompt) -> GenerationResult:
        """Generate conversation using Amazon Bedrock Converse API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "messages": [{"role": "user", "content": [{"text": prompt.user_prompt}]}],
            "system": [{"text": prompt.system_prompt}],
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self.endpoint, json=body, headers=headers)
                response.raise_for_status()
                data = response.json()

            response_text = data["output"]["message"]["content"][0]["text"]

            conversation = self._parse_json_response(response_text)

            if conversation is None:
                return GenerationResult(
                    success=False,
                    conversation=None,
                    raw_response=response_text,
                    error="Failed to parse JSON from response",
                )

            if "messages" not in conversation:
                return GenerationResult(
                    success=False,
                    conversation=None,
                    raw_response=response_text,
                    error="Response missing 'messages' field",
                )

            return GenerationResult(
                success=True,
                conversation=conversation,
                raw_response=response_text,
                error=None,
            )

        except httpx.HTTPStatusError as e:
            return GenerationResult(
                success=False,
                conversation=None,
                raw_response=e.response.text,
                error=f"Bedrock API Error {e.response.status_code}: {e.response.text}",
            )
        except Exception as e:
            return GenerationResult(
                success=False,
                conversation=None,
                raw_response="",
                error=f"Error: {str(e)}",
            )
