"""
Tests for MCP server construction.

Covers both the pre-2.0 decorator-based ``Server`` API and the 2.0+
constructor-callback API, since ``mcp`` made this a breaking change
between major versions and ``_build_server`` supports both.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import pytest

from mcp import types

from fiftyone_mcp import server as server_mod
from fiftyone_mcp.registry import ToolRegistry


def _make_registry():
    registry = ToolRegistry()
    schema = types.Tool(
        name="ping",
        description="ping tool",
        inputSchema={"type": "object", "properties": {}},
    )
    registry.register(schema, lambda ctx, **kwargs: {"success": True})
    return registry


class TestBuildServerInstalledAPI:
    """Exercises ``_build_server`` against whichever ``mcp`` API is
    actually installed in this environment."""

    def test_builds_and_initializes(self):
        registry = _make_registry()
        server = server_mod._build_server("fiftyone-mcp", registry)
        opts = server.create_initialization_options()
        assert opts.server_name == "fiftyone-mcp"


class _FakeServerV2:
    """Stands in for ``mcp`` 2.0+'s ``Server``: no decorator methods;
    handlers are passed as constructor callbacks instead."""

    def __init__(self, name, *, on_list_tools=None, on_call_tool=None):
        self.name = name
        self.on_list_tools = on_list_tools
        self.on_call_tool = on_call_tool


class _CallToolParams:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class TestBuildServerV2API:
    """Verifies the mcp>=2.0 branch without requiring mcp 2.0 to be
    installed, by swapping in a fake Server with the 2.0 shape."""

    def test_passes_handlers_as_constructor_kwargs(self, monkeypatch):
        monkeypatch.setattr(server_mod, "Server", _FakeServerV2)
        registry = _make_registry()

        server = server_mod._build_server("fiftyone-mcp", registry)

        assert isinstance(server, _FakeServerV2)
        assert server.on_list_tools is not None
        assert server.on_call_tool is not None

    @pytest.mark.asyncio
    async def test_on_list_tools_returns_typed_result(self, monkeypatch):
        monkeypatch.setattr(server_mod, "Server", _FakeServerV2)
        registry = _make_registry()
        server = server_mod._build_server("fiftyone-mcp", registry)

        result = await server.on_list_tools(None, None)

        assert isinstance(result, types.ListToolsResult)
        assert result.tools[0].name == "ping"

    @pytest.mark.asyncio
    async def test_on_call_tool_returns_typed_result(self, monkeypatch):
        monkeypatch.setattr(server_mod, "Server", _FakeServerV2)
        registry = _make_registry()
        server = server_mod._build_server("fiftyone-mcp", registry)

        result = await server.on_call_tool(
            None, _CallToolParams("ping", {})
        )

        assert isinstance(result, types.CallToolResult)
        assert '"success": true' in result.content[0].text.lower()
