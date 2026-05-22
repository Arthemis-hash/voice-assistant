"""
LM Studio LLM provider.

Connects to LM Studio's OpenAI-compatible HTTP API.
Supports both sync and streaming responses.
"""

import logging
from typing import Generator, Optional

import requests

logger = logging.getLogger(__name__)


class LMStudioProvider:
    """
    LLM provider for LM Studio HTTP server.

    LM Studio serves local models via OpenAI-compatible API.
    Default endpoint: http://localhost:1234/v1

    Usage:
        llm = LMStudioProvider(
            base_url="http://localhost:1234/v1",
            model="Qwen3-4B",
        )

        response = llm.generate("Hello, how are you?")
        print(response)

        # Streaming
        for chunk in llm.generate_stream("Tell me a story"):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "Qwen3-4B",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._session = requests.Session()

    def _make_request(
        self,
        messages: list[dict],
        stream: bool = False,
    ) -> requests.Response:
        """Make API request to LM Studio."""
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

        response = self._session.post(
            url,
            json=payload,
            timeout=self.timeout,
            stream=stream,
        )
        response.raise_for_status()
        return response

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a text response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._make_request(messages, stream=False)
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to LM Studio at {self.base_url}. "
                f"Is the server running?"
            ) from e
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Generate a streaming text response from the LLM.

        Yields text chunks as they're generated.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Yields:
            Text chunks
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._make_request(messages, stream=True)

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break

                    import json
                    data = json.loads(data_str)
                    if data.get("choices"):
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content

        except requests.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to LM Studio at {self.base_url}. "
                f"Is the server running?"
            ) from e
        except Exception as e:
            raise RuntimeError(f"LLM streaming request failed: {e}") from e

    def is_available(self) -> bool:
        """Check if LM Studio server is reachable."""
        try:
            response = self._session.get(
                f"{self.base_url}/models",
                timeout=5.0,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        """List available models from LM Studio."""
        try:
            response = self._session.get(
                f"{self.base_url}/models",
                timeout=5.0,
            )
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except requests.RequestException:
            return []
