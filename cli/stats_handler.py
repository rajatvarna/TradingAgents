import threading
from typing import Any, Dict, List, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import AIMessage


class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, and token usage."""

    def __init__(self, provider: str = "google") -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.provider = provider.lower()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.total_cost = 0.0

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when an LLM starts."""
        with self._lock:
            self.llm_calls += 1

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when a chat model starts."""
        with self._lock:
            self.llm_calls += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract token usage from LLM response."""
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage_metadata = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                usage_metadata = message.usage_metadata

        if usage_metadata:
            with self._lock:
                t_in = usage_metadata.get("input_tokens", 0)
                t_out = usage_metadata.get("output_tokens", 0)
                self.tokens_in += t_in
                self.tokens_out += t_out
                
                # Simple pricing estimate (per 1M tokens) based on late 2024 pricing
                if self.provider == "google":
                    self.total_cost += (t_in / 1_000_000 * 1.25) + (t_out / 1_000_000 * 5.00)
                elif self.provider == "openai":
                    self.total_cost += (t_in / 1_000_000 * 2.50) + (t_out / 1_000_000 * 10.00)
                elif self.provider == "anthropic":
                    self.total_cost += (t_in / 1_000_000 * 3.00) + (t_out / 1_000_000 * 15.00)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Increment tool call counter when a tool starts."""
        with self._lock:
            self.tool_calls += 1

    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "total_cost": self.total_cost,
            }
