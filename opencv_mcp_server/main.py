#!/usr/bin/env python3
"""
Merlin CV MCP Server - Standardized entrypoint.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import logging
import os
import sys
from ipaddress import ip_address

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import uvicorn

# Import shared MCP instance
from .mcp_instance import mcp
# Import modules to trigger tool registration via decorators
from . import computer_vision, image_basics, image_processing, security, video_processing, profile_tool

from .utils.config import DEFAULT_HOST, DEFAULT_PORT
from .security import record_security_event

logger = logging.getLogger("merlin-cv-mcp")

DEFAULT_AUTH_TOKEN_ENV_VAR = "MCP_MERLIN_AUTH_TOKEN"

def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

def _is_loopback_host(host: str) -> bool:
    normalized = (host or "").strip().lower()
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False

def _extract_bearer_token(auth_header: str | None) -> str | None:
    if not auth_header:
        return None
    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None

class _AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Starlette, token: str):
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):
        bearer = _extract_bearer_token(request.headers.get("authorization"))
        header_token = request.headers.get("x-mcp-auth-token")
        candidate = bearer or (header_token.strip() if header_token else None)

        if candidate and hmac.compare_digest(candidate, self._token):
            return await call_next(request)

        record_security_event(
            "http_auth_failed",
            path=request.url.path,
            method=request.method,
            remote=getattr(request.client, "host", None),
        )
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

def _create_http_app(mode: str, auth_token: str | None) -> Starlette:
    if mode == "sse":
        app = mcp.sse_app()
    elif mode == "streamable-http":
        app = mcp.streamable_http_app()
    else:
        raise ValueError(f"Unsupported HTTP mode: {mode}")

    if auth_token:
        app.add_middleware(_AuthMiddleware, token=auth_token)
    return app

async def run_server(
    *,
    mode: str,
    host: str,
    port: int,
    allow_remote_http: bool,
    auth_token: str | None,
) -> None:
    if mode == "stdio":
        logger.info("Starting Merlin CV MCP in stdio mode")
        await mcp.run_stdio_async()
        return

    if mode not in {"sse", "streamable-http"}:
        raise ValueError(f"Unknown mode: {mode}")

    loopback_host = _is_loopback_host(host)
    if not loopback_host and not allow_remote_http:
        record_security_event("remote_bind_blocked", host=host, reason="missing_allow_remote_http")
        raise ValueError("Refusing to bind HTTP transport to a non-loopback host without --allow-remote-http.")

    if not loopback_host and not auth_token:
        record_security_event("remote_bind_blocked", host=host, reason="missing_auth_token")
        raise ValueError("Remote HTTP binding requires an auth token.")

    http_app = _create_http_app(mode=mode, auth_token=auth_token)
    config = uvicorn.Config(http_app, host=host, port=port, log_level="info")
    uvicorn_server = uvicorn.Server(config)

    logger.info("Starting Merlin CV MCP in %s mode on %s:%s", mode, host, port)
    await uvicorn_server.serve()

async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Merlin CV MCP Server")
    parser.add_argument("--mode", choices=["stdio", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--allow-remote-http", action="store_true")
    parser.add_argument("--auth-token-env", default=DEFAULT_AUTH_TOKEN_ENV_VAR)
    args = parser.parse_args()

    port = args.port if args.port is not None else int(os.environ.get("PORT", DEFAULT_PORT))
    auth_token = os.environ.get(args.auth_token_env, "").strip() or None

    await run_server(
        mode=args.mode,
        host=args.host,
        port=port,
        allow_remote_http=args.allow_remote_http,
        auth_token=auth_token,
    )

def main() -> None:
    _configure_logging()
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
