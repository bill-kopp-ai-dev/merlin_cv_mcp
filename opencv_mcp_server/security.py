"""
Runtime security telemetry for Merlin CV MCP.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger("opencv-mcp-server.security")

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


def reset_security_metrics_for_tests() -> None:
    with _COUNTERS_LOCK:
        _SECURITY_COUNTERS.clear()


def get_security_metrics() -> Dict[str, Any]:
    """
    Retorna métricas de segurança em memória desde o início do processo.

    USO ESTRATÉGICO NO AGENTE:
    - Chame após erro de permissão/autenticação para saber se foi bloqueio de política
      (ex: `workspace_path_blocked`, `remote_bind_blocked`, `camera_access_blocked`).
    - Útil para auditoria rápida em produção sem reiniciar o servidor.

    CAMPOS PRINCIPAIS:
    - `security_metrics`: contador por tipo de evento.
    - `captured_events`: total agregado de eventos registrados.
    - `updated_at_utc`: timestamp UTC da leitura.

    IMPORTANTE:
    - Métricas são voláteis (memória local) e resetam ao reiniciar o processo.
    """
    metrics = get_security_metrics_snapshot()
    return {
        "security_metrics": metrics,
        "captured_events": int(sum(metrics.values())),
        "updated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }


def register_tools(mcp) -> None:
    mcp.add_tool(get_security_metrics)
