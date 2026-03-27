"""LLM module using Groq Cloud for chat completions."""

import logging

from groq import Groq

from src.config import GroqConfig

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Custom exception for LLM errors."""


class GroqLLM:
    """LLM chat completion using the Groq Cloud API.

    Args:
        config: Groq configuration containing API key and model settings.
    """

    def __init__(self, config: GroqConfig) -> None:
        """Initialise the Groq LLM client.

        Args:
            config: Groq configuration containing API key and model settings.

        Raises:
            LLMError: If the API key is missing.
        """
        if not config.api_key:
            raise LLMError("Groq API key is required")
        self._config = config
        self._client = Groq(api_key=config.api_key)
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        logger.info("Initialised GroqLLM with model %s", config.model)

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate a response for the given prompt.

        Args:
            prompt: The user prompt text.
            system_prompt: Optional system-level instruction.

        Returns:
            The generated text response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
        except self._client.AuthenticationError as exc:
            raise LLMError("Groq authentication failed: " + str(exc)) from exc
        except self._client.APIError as exc:
            raise LLMError("Groq API error: " + str(exc)) from exc
        self.call_count += 1
        usage = response.usage
        if usage:
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens
        answer = response.choices[0].message.content or ""
        logger.debug(
            "Generated response (%d input tokens, %d output tokens)",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
        return answer

    def generate_with_context(self, query: str, context: str, system_prompt: str = None) -> str:
        """Generate a response using retrieved context and a user query.

        Builds a prompt that presents the context first, followed by the
        user's question.

        Args:
            query: The user's question.
            context: Retrieved context text.
            system_prompt: Optional system-level instruction.

        Returns:
            The generated text response.
        """
        prompt = "Context:\n" + context + "\n\nUser Question: " + query + "\n\nPlease answer the user's question based on the context provided above."
        return self.generate(prompt, system_prompt=system_prompt)
