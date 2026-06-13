import json
import logging
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


def _extract_fingerprint(response: LLMResult) -> dict[str, str | None]:
    """Best-effort extraction of model identity + fingerprint from an LLM result.

    Returns ``{"model": str|None, "fingerprint": str|None}``. We look in a few
    places because LangChain exposes provider metadata inconsistently:
    - ``response.llm_output`` is the OpenAI Chat-Completions convention and
      carries ``system_fingerprint`` directly.
    - ``generation.generation_info`` is used by some providers (incl. OpenAI
      Responses API; Anthropic surfaces the model id here as ``model``).
    - ``message.response_metadata`` is the modern home for both fields on
      chat-model providers.

    The presence/absence of ``fingerprint`` is itself audit signal — record it
    even when None, so drift detection can distinguish "provider didn't return one"
    from "we forgot to look".
    """
    model: str | None = None
    fingerprint: str | None = None

    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        model = model or llm_output.get("model_name") or llm_output.get("model")
        fingerprint = fingerprint or llm_output.get("system_fingerprint")

    try:
        generation = response.generations[0][0]
    except (IndexError, TypeError, AttributeError):
        generation = None

    if generation is not None:
        gen_info = getattr(generation, "generation_info", None) or {}
        if isinstance(gen_info, dict):
            model = model or gen_info.get("model_name") or gen_info.get("model")
            fingerprint = fingerprint or gen_info.get("system_fingerprint")

        message = getattr(generation, "message", None)
        meta = getattr(message, "response_metadata", None) or {}
        if isinstance(meta, dict):
            model = model or meta.get("model_name") or meta.get("model")
            fingerprint = fingerprint or meta.get("system_fingerprint")

    return {"model": model, "fingerprint": fingerprint}


class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, and token usage."""

    def __init__(
        self,
        provider: str = "google",
        *,
        jsonl_path: str | Path | None = None,
        session_id: str | None = None,
    ) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.provider = provider.lower()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.total_cost = 0.0

        # Per-call identity records
        self.fingerprints: list[dict[str, str | None]] = []

        # JSONL persistence
        self.session_id: str = session_id or str(uuid.uuid4())
        self.jsonl_path: Path | None = (
            Path(jsonl_path).expanduser() if jsonl_path else None
        )
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        self._call_seq = 0

        # Role-specific stats
        self.role_stats: dict[str, dict[str, int | float]] = {}
        self._run_context: dict[str, dict[str, str | float]] = {}

    @staticmethod
    def _normalize_role(raw_role: str) -> str:
        role = (raw_role or "unknown").strip().lower().replace("-", " ")
        compact = " ".join(role.split())
        mapping = {
            "market analyst": "market_analyst",
            "social analyst": "social_media_analyst",
            "social media analyst": "social_media_analyst",
            "news analyst": "news_analyst",
            "fundamentals analyst": "fundamentals_analyst",
            "bull researcher": "bull_researcher",
            "bear researcher": "bear_researcher",
            "research manager": "research_manager",
            "trader": "trader",
            "aggressive analyst": "aggressive_analyst",
            "neutral analyst": "neutral_analyst",
            "conservative analyst": "conservative_analyst",
            "portfolio manager": "portfolio_manager",
        }
        return mapping.get(compact, compact.replace(" ", "_") or "unknown")

    @classmethod
    def _extract_role(
        cls,
        serialized: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        metadata = kwargs.get("metadata") or {}
        if isinstance(metadata, dict):
            for key in ("langgraph_node", "graph_node", "node", "node_name"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return cls._normalize_role(value)

        tags = kwargs.get("tags") or []
        if isinstance(tags, list):
            for tag in tags:
                if not isinstance(tag, str):
                    continue
                if tag.startswith("role:"):
                    return cls._normalize_role(tag.split(":", 1)[1])

        name = serialized.get("name") if isinstance(serialized, dict) else None
        if isinstance(name, str) and name.strip():
            return cls._normalize_role(name)

        return "unknown"

    @staticmethod
    def _run_id_from_kwargs(**kwargs: Any) -> str:
        run_id = kwargs.get("run_id")
        return str(run_id) if run_id is not None else ""

    def _ensure_role_bucket(self, role: str) -> dict[str, int | float]:
        if role not in self.role_stats:
            self.role_stats[role] = {
                "llm_calls": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "total_tokens": 0,
                "total_duration_seconds": 0.0,
            }
        return self.role_stats[role]

    def _register_start(self, serialized: dict[str, Any], **kwargs: Any) -> None:
        role = self._extract_role(serialized, **kwargs)
        run_id = self._run_id_from_kwargs(**kwargs)
        started_at = time.perf_counter()

        with self._lock:
            self.llm_calls += 1
            role_bucket = self._ensure_role_bucket(role)
            role_bucket["llm_calls"] = int(role_bucket["llm_calls"]) + 1
            if run_id:
                self._run_context[run_id] = {
                    "role": role,
                    "started_at": started_at,
                }

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when an LLM starts."""
        if run_id is not None and "run_id" not in kwargs:
            kwargs = {**kwargs, "run_id": run_id}
        self._register_start(serialized, **kwargs)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when a chat model starts."""
        if run_id is not None and "run_id" not in kwargs:
            kwargs = {**kwargs, "run_id": run_id}
        self._register_start(serialized, **kwargs)

    def on_llm_end(
        self,
        response: LLMResult,
        run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Extract token usage, fingerprint, and write a per-call JSONL line."""
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage_metadata = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                usage_metadata = message.usage_metadata

        identity = _extract_fingerprint(response)
        tokens_in = (usage_metadata or {}).get("input_tokens", 0) or 0
        tokens_out = (usage_metadata or {}).get("output_tokens", 0) or 0
        total_tokens = tokens_in + tokens_out

        r_id = str(run_id) if run_id is not None else self._run_id_from_kwargs(**kwargs)

        with self._lock:
            role = "unknown"
            duration = 0.0
            started_at = 0.0
            if r_id and r_id in self._run_context:
                ctx = self._run_context.pop(r_id)
                role = str(ctx.get("role") or "unknown")
                started_at = float(ctx.get("started_at") or 0.0)
                if started_at > 0.0:
                    duration = max(0.0, time.perf_counter() - started_at)

            role_bucket = self._ensure_role_bucket(role)
            role_bucket["total_duration_seconds"] = float(role_bucket["total_duration_seconds"]) + duration

            self.tokens_in += tokens_in
            self.tokens_out += tokens_out

            role_bucket["tokens_in"] = int(role_bucket["tokens_in"]) + tokens_in
            role_bucket["tokens_out"] = int(role_bucket["tokens_out"]) + tokens_out
            role_bucket["total_tokens"] = int(role_bucket["total_tokens"]) + total_tokens

            # Simple pricing estimate (per 1M tokens) based on late 2024 pricing
            if self.provider == "google":
                self.total_cost += (tokens_in / 1_000_000 * 1.25) + (tokens_out / 1_000_000 * 5.00)
            elif self.provider == "openai":
                self.total_cost += (tokens_in / 1_000_000 * 2.50) + (tokens_out / 1_000_000 * 10.00)
            elif self.provider == "anthropic":
                self.total_cost += (tokens_in / 1_000_000 * 3.00) + (tokens_out / 1_000_000 * 15.00)

            self.fingerprints.append(identity)
            self._call_seq += 1
            seq = self._call_seq

            latency_ms = int(duration * 1000) if started_at > 0.0 else None

            if self.jsonl_path is not None:
                record = {
                    "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "type": "llm_end",
                    "session_id": self.session_id,
                    "call_seq": seq,
                    "call_id": r_id if r_id else None,
                    "model": identity["model"],
                    "fingerprint": identity["fingerprint"],
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "latency_ms": latency_ms,
                    "tags": list(tags) if tags else [],
                    "metadata": dict(metadata) if isinstance(metadata, dict) else {},
                }
                try:
                    with open(self.jsonl_path, "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(record, ensure_ascii=False, default=str) + "\n"
                        )
                except Exception as e:
                    logger.warning("audit jsonl write failed: %s", e)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Track duration for failed runs so latency averages stay representative."""
        run_id = self._run_id_from_kwargs(**kwargs)
        if not run_id:
            return
        with self._lock:
            ctx = self._run_context.pop(run_id, None)
            if not ctx:
                return
            role = str(ctx.get("role") or "unknown")
            started_at = float(ctx.get("started_at") or 0.0)
            duration = max(0.0, time.perf_counter() - started_at) if started_at > 0.0 else 0.0
            role_bucket = self._ensure_role_bucket(role)
            role_bucket["total_duration_seconds"] = float(role_bucket["total_duration_seconds"]) + duration

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Increment tool call counter when a tool starts."""
        with self._lock:
            self.tool_calls += 1

    # ------------------------------------------------------------------ #
    # Public read API
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict[str, Any]:
        """Return current statistics."""
        with self._lock:
            role_stats = {}
            for role, stats in self.role_stats.items():
                calls = int(stats.get("llm_calls", 0) or 0)
                duration = float(stats.get("total_duration_seconds", 0.0) or 0.0)
                role_stats[role] = {
                    "llm_calls": calls,
                    "tokens_in": int(stats.get("tokens_in", 0) or 0),
                    "tokens_out": int(stats.get("tokens_out", 0) or 0),
                    "total_tokens": int(stats.get("total_tokens", 0) or 0),
                    "total_duration_seconds": duration,
                    "avg_duration_seconds": (duration / calls) if calls else 0.0,
                }
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "total_cost": self.total_cost,
                "fingerprints": list(self.fingerprints),
                "session_id": self.session_id,
                "jsonl_path": str(self.jsonl_path) if self.jsonl_path else None,
                "role_stats": role_stats,
            }
