"""
FiftyOne MCP Server.

Main entrypoint for the FiftyOne Model Context Protocol server.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import asyncio
import json
import logging
from pathlib import Path

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .registry import ToolRegistry
from .tools import (
    aggregations,
    app_config,
    datasets,
    evaluation,
    operations,
    operators,
    pipelines,
    plugins,
    samples,
    schema,
    session,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_registry(config=None):
    """Builds the tool registry with all MCP tools.

    Args:
        config (None): optional config dict from ``settings.json``

    Returns:
        a :class:`ToolRegistry` instance with all tools registered
    """
    server_config = (config or {}).get("server", {})
    max_response_chars = server_config.get("max_response_chars", None)

    registry = (
        ToolRegistry(max_response_chars=max_response_chars)
        if max_response_chars is not None
        else ToolRegistry()
    )
    aggregations.register_tools(registry)
    app_config.register_tools(registry)
    datasets.register_tools(registry)
    evaluation.register_tools(registry)
    operations.register_tools(registry)
    operators.register_tools(registry)
    pipelines.register_tools(registry)
    plugins.register_tools(registry)
    samples.register_tools(registry)
    schema.register_tools(registry)
    session.register_tools(registry)
    return registry


def load_config():
    """Loads configuration from settings.json.

    Returns:
        a config dict
    """
    config_path = Path(__file__).parent / "config" / "settings.json"
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load config from %s: %s", config_path, e)
        return {}


def _build_server(server_name, registry):
    """Constructs and wires up the MCP ``Server`` instance.

    ``mcp`` 2.0 replaced the pre-2.0 decorator-based handler
    registration (``@server.list_tools()`` / ``@server.call_tool()``)
    with constructor callbacks (``on_list_tools`` / ``on_call_tool``)
    that must return typed ``mcp.types`` result objects. Both are
    supported here, selected via a capability check, since
    ``pyproject.toml`` allows both major lines.

    Args:
        server_name: the server's display name
        registry: a :class:`~fiftyone_mcp.registry.ToolRegistry`
            instance

    Returns:
        a configured :class:`mcp.server.Server` instance
    """
    if hasattr(Server, "list_tools"):
        server = Server(server_name)

        @server.list_tools()
        async def _list_tools_handler():
            return registry.list_tools()

        @server.call_tool()
        async def _call_tool_handler(name, arguments):
            result = await registry.call_tool(name, arguments, ctx=None)
            return result.content

        return server

    async def _on_list_tools(ctx, params):
        return types.ListToolsResult(tools=registry.list_tools())

    async def _on_call_tool(ctx, params):
        result = await registry.call_tool(
            params.name, params.arguments, ctx=None
        )
        return types.CallToolResult(content=result.content)

    return Server(  # pylint: disable=unexpected-keyword-arg
        server_name,
        on_list_tools=_on_list_tools,
        on_call_tool=_on_call_tool,
    )


async def main():
    """Main server function."""
    config = load_config()
    server_config = config.get("server", {})
    server_name = server_config.get("name", "fiftyone-mcp")

    logger.info("Starting %s server...", server_name)

    registry = build_registry(config=config)
    server = _build_server(server_name, registry)

    logger.info("%s server initialized successfully", server_name)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run():
    """Entry point for the server."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error("Server error: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    run()
