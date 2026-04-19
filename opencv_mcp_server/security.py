from __future__ import annotations

import logging
import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict

from .mcp_instance import mcp
from .utils.contracts import success_response

logger = logging.getLogger("merlin-cv-mcp.security")

_COUNTERS_LOCK = threading.Lock()
_SECURITY_COUNTERS: Counter[str] = Counter()

def _sanitize_detail_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if len(text) > 200:
        text = f"{text[:200]}..."
    return text

def record_security_event(event: str, **details: Any) -> None:
    event_name = str(event).strip() or "unknown_security_event"
    with _COUNTERS_LOCK:
        _SECURITY_COUNTERS[event_name] += 1

    if details:
        safe_details = {k: _sanitize_detail_value(v) for k, v in details.items()}
        logger.warning("Security event: %s | details=%s", event_name, safe_details)
    else:
        logger.warning("Security event: %s", event_name)

def get_security_metrics_snapshot() -> Dict[str, int]:
    with _COUNTERS_LOCK:
        return dict(_SECURITY_COUNTERS)

@mcp.tool()
async def get_security_metrics() -> Dict[str, Any]:
    """
    Returns real-time security metrics since the process started.
    
    Fields:
    - security_metrics: counters by event type.
    - captured_events: total count.
    - updated_at_utc: ISO timestamp.
    """
    metrics = get_security_metrics_snapshot()
    data = {
        "security_metrics": metrics,
        "captured_events": int(sum(metrics.values())),
        "updated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    return success_response(data, tool_name="get_security_metrics")
