from mcp.server.fastmcp import FastMCP
from .utils.config import SERVER_NAME, DEFAULT_HOST, DEFAULT_PORT

# Global MCP instance to be used across modules for decorators
mcp = FastMCP(
    SERVER_NAME,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    sse_path="/sse",
    message_path="/messages/",
    streamable_http_path="/mcp",
    stateless_http=False,
)
