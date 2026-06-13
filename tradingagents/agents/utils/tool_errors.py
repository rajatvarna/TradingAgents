import json
from typing import Any


def tool_error_payload(*, tool: str, error: Exception, vendor: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": True,
        "tool": tool,
        "type": type(error).__name__,
        "message": str(error),
    }
    if vendor is not None:
        payload["vendor"] = vendor
    return payload


def tool_error_text(*, tool: str, error: Exception, vendor: str | None = None) -> str:
    return json.dumps(tool_error_payload(tool=tool, error=error, vendor=vendor), ensure_ascii=False)
