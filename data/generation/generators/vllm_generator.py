"""Local vLLM generator with constrained JSON decoding (xgrammar).

Run vLLM on the H100 before generating:

    vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.92 \
        --enable-chunked-prefill \
        --max-num-batched-tokens 8192 \
        --port 8000

Expected throughput: ~4 000-5 000 tok/s with 100 concurrent workers.
Swap to Qwen2.5-7B-Instruct for ~18 000 tok/s at lower quality.
"""

import os
from typing import Literal, Optional

import httpx
from pydantic import BaseModel

from data.generation.generators.base import ConversationGenerator, GenerationResult
from data.generation.prompts.base import ConversationPrompt


class _MessageOut(BaseModel):
    role: Literal["sent", "received"]
    content: str


class _ConversationOut(BaseModel):
    messages: list[_MessageOut]
    reasoning: str


_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ConversationOut",
        "schema": _ConversationOut.model_json_schema(),
        "strict": True,
    },
}

_DEFAULT_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class VLLMGenerator(ConversationGenerator):
    """Generate conversations using a local vLLM server with constrained JSON decoding."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: Optional[str] = None,
        temperature: float = 0.9,
        max_tokens: int = 512,
        timeout: float = 120.0,
        max_connections: int = 200,
    ):
        self.model = model
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._endpoint = f"{self.base_url}/chat/completions"
        # Single persistent client shared across all requests — avoids opening
        # a new TCP connection per request which exhausts OS limits at high concurrency
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
        self._client = httpx.AsyncClient(timeout=timeout, limits=limits)

    @property
    def name(self) -> str:
        return "vllm"

    async def generate(self, prompt: ConversationPrompt) -> GenerationResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": _RESPONSE_FORMAT,
        }

        try:
            response = await self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            conversation = self._parse_json_response(text)

            if conversation is None:
                return GenerationResult(
                    success=False,
                    conversation=None,
                    raw_response=text,
                    error="Failed to parse JSON from response",
                )

            if "messages" not in conversation:
                return GenerationResult(
                    success=False,
                    conversation=None,
                    raw_response=text,
                    error="Response missing 'messages' field",
                )

            return GenerationResult(success=True, conversation=conversation, raw_response=text)

        except httpx.HTTPStatusError as e:
            return GenerationResult(
                success=False,
                conversation=None,
                raw_response=e.response.text,
                error=f"vLLM HTTP {e.response.status_code}: {e.response.text[:200]}",
            )
        except Exception as e:
            return GenerationResult(
                success=False,
                conversation=None,
                raw_response="",
                error=str(e),
            )
