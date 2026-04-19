from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional, TypedDict

class ToolResponse(TypedDict, total=False):
    ok: bool
    data: Optional[Any]
    error: Optional[str]
    code: Optional[str]
    details: Optional[Dict[str, Any]]
    meta: Dict[str, Any]
    request_id: str

def create_response(
    ok: bool,
    data: Optional[Any] = None,
    error: Optional[str] = None,
    code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    tool_name: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    rid = request_id or str(uuid.uuid4())
    resp: ToolResponse = {
        "ok": ok,
        "request_id": rid,
        "meta": {
            "server": "merlin-cv-mcp",
            "timestamp": time.time(),
            "tool": tool_name
        }
    }
    
    if ok:
        resp["data"] = data
    else:
        resp["error"] = error or "Unknown error"
        resp["code"] = code or "internal_error"
        if details:
            resp["details"] = details
            
    return resp

def success_response(data: Any, tool_name: Optional[str] = None, request_id: Optional[str] = None) -> ToolResponse:
    return create_response(True, data=data, tool_name=tool_name, request_id=request_id)

def error_response(error: str, code: str = "internal_error", details: Optional[Dict[str, Any]] = None, tool_name: Optional[str] = None, request_id: Optional[str] = None) -> ToolResponse:
    return create_response(False, error=error, code=code, details=details, tool_name=tool_name, request_id=request_id)
