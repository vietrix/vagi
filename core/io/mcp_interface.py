#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Interface for vAGI.

Implements the MCP standard for tool discovery and execution, allowing vAGI
to function as an MCP Host that can interact with external tools and services.

MCP Protocol Overview:
    - JSON-RPC 2.0 based communication
    - Tools are discovered via `tools/list` method
    - Tools are invoked via `tools/call` method
    - Supports streaming responses for long-running operations

Reference: https://modelcontextprotocol.io/

Usage:
    from core.io.mcp_interface import MCPServer, FileSystemTool

    # Create server with tools
    server = MCPServer()
    server.register_tool(FileSystemTool(allowed_paths=["/data"]))

    # Handle JSON-RPC request
    response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list"
    })

    # Call a tool
    response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "filesystem.read_file",
            "arguments": {"path": "/data/example.txt"}
        }
    })
"""

from __future__ import annotations

import json
import logging
import os
import re
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ============================================================================
# MCP Protocol Types
# ============================================================================

class MCPErrorCode(IntEnum):
    """JSON-RPC error codes following MCP specification."""
    # Standard JSON-RPC errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # MCP-specific errors
    TOOL_NOT_FOUND = -32000
    TOOL_EXECUTION_ERROR = -32001
    PERMISSION_DENIED = -32002
    RESOURCE_NOT_FOUND = -32003
    RATE_LIMITED = -32004
    TIMEOUT = -32005


@dataclass
class MCPError(Exception):
    """MCP protocol error."""
    code: MCPErrorCode
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-RPC error object."""
        error = {
            "code": self.code.value,
            "message": self.message,
        }
        if self.data is not None:
            error["data"] = self.data
        return error


@dataclass
class MCPToolParameter:
    """Parameter definition for an MCP tool."""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None


@dataclass
class MCPToolDefinition:
    """Definition of an MCP tool for discovery."""
    name: str
    description: str
    parameters: List[MCPToolParameter] = field(default_factory=list)

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format for tools/list response."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }


@dataclass
class MCPToolResult:
    """Result of an MCP tool execution."""
    content: List[Dict[str, Any]]
    is_error: bool = False

    @classmethod
    def text(cls, text: str) -> "MCPToolResult":
        """Create a text result."""
        return cls(content=[{"type": "text", "text": text}])

    @classmethod
    def json_data(cls, data: Any) -> "MCPToolResult":
        """Create a JSON result."""
        return cls(content=[{"type": "text", "text": json.dumps(data, indent=2, ensure_ascii=False)}])

    @classmethod
    def error(cls, message: str) -> "MCPToolResult":
        """Create an error result."""
        return cls(content=[{"type": "text", "text": f"Error: {message}"}], is_error=True)

    @classmethod
    def image(cls, data: str, mime_type: str = "image/png") -> "MCPToolResult":
        """Create an image result (base64)."""
        return cls(content=[{"type": "image", "data": data, "mimeType": mime_type}])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP response format."""
        return {
            "content": self.content,
            "isError": self.is_error,
        }


# ============================================================================
# MCP Tool Base Class
# ============================================================================

class MCPTool(ABC):
    """
    Abstract base class for MCP tools.

    Implement this class to create custom tools that can be registered
    with the MCP server.

    Example:
        class MyTool(MCPTool):
            @property
            def name(self) -> str:
                return "my_tool"

            @property
            def description(self) -> str:
                return "Does something useful"

            def get_parameters(self) -> List[MCPToolParameter]:
                return [
                    MCPToolParameter("input", "string", "Input value", required=True)
                ]

            def execute(self, arguments: Dict[str, Any]) -> MCPToolResult:
                return MCPToolResult.text(f"Got: {arguments['input']}")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the tool (e.g., 'filesystem.read_file')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass

    @abstractmethod
    def get_parameters(self) -> List[MCPToolParameter]:
        """Return list of parameters this tool accepts."""
        pass

    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> MCPToolResult:
        """
        Execute the tool with given arguments.

        Args:
            arguments: Dictionary of parameter name -> value

        Returns:
            MCPToolResult with execution output

        Raises:
            MCPError: On execution failure
        """
        pass

    def get_definition(self) -> MCPToolDefinition:
        """Get tool definition for discovery."""
        return MCPToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.get_parameters(),
        )

    def validate_arguments(self, arguments: Dict[str, Any]) -> None:
        """
        Validate arguments against parameter definitions.

        Raises MCPError if validation fails.
        """
        params = {p.name: p for p in self.get_parameters()}

        # Check required parameters
        for name, param in params.items():
            if param.required and name not in arguments:
                raise MCPError(
                    MCPErrorCode.INVALID_PARAMS,
                    f"Missing required parameter: {name}"
                )

        # Check for unknown parameters
        for name in arguments:
            if name not in params:
                raise MCPError(
                    MCPErrorCode.INVALID_PARAMS,
                    f"Unknown parameter: {name}"
                )

        # Type validation (basic)
        for name, value in arguments.items():
            param = params[name]
            if param.enum and value not in param.enum:
                raise MCPError(
                    MCPErrorCode.INVALID_PARAMS,
                    f"Parameter '{name}' must be one of: {param.enum}"
                )


# ============================================================================
# MCP Server
# ============================================================================

class MCPServer:
    """
    MCP Server implementation for vAGI.

    Handles JSON-RPC 2.0 requests following the MCP specification.
    Supports tool discovery and execution.

    Protocol Methods:
        - tools/list: List available tools
        - tools/call: Execute a tool

    Usage:
        server = MCPServer()
        server.register_tool(FileSystemTool())
        server.register_tool(CalculatorTool())

        # Handle incoming request
        request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        response = server.handle_request(request)
    """

    JSONRPC_VERSION = "2.0"
    MCP_VERSION = "2024-11-05"

    def __init__(
        self,
        name: str = "vAGI MCP Server",
        version: str = "1.0.0",
    ):
        """
        Initialize MCP server.

        Args:
            name: Server name for identification
            version: Server version
        """
        self.name = name
        self.version = version
        self.tools: Dict[str, MCPTool] = {}
        self._request_count = 0

    def register_tool(self, tool: MCPTool) -> None:
        """
        Register a tool with the server.

        Args:
            tool: MCPTool instance to register

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool already registered: {tool.name}")

        self.tools[tool.name] = tool
        logger.info(f"Registered MCP tool: {tool.name}")

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool by name."""
        if name in self.tools:
            del self.tools[name]
            logger.info(f"Unregistered MCP tool: {name}")

    def handle_request(self, request: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Handle an incoming JSON-RPC request.

        Args:
            request: JSON-RPC request (string or dict)

        Returns:
            JSON-RPC response dict
        """
        self._request_count += 1

        # Parse if string
        if isinstance(request, str):
            try:
                request = json.loads(request)
            except json.JSONDecodeError as e:
                return self._error_response(
                    None,
                    MCPError(MCPErrorCode.PARSE_ERROR, f"Invalid JSON: {e}")
                )

        # Validate request structure
        if not isinstance(request, dict):
            return self._error_response(
                None,
                MCPError(MCPErrorCode.INVALID_REQUEST, "Request must be an object")
            )

        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        # Validate JSON-RPC version
        if request.get("jsonrpc") != self.JSONRPC_VERSION:
            return self._error_response(
                request_id,
                MCPError(MCPErrorCode.INVALID_REQUEST, f"Must be JSON-RPC {self.JSONRPC_VERSION}")
            )

        if not method:
            return self._error_response(
                request_id,
                MCPError(MCPErrorCode.INVALID_REQUEST, "Method is required")
            )

        # Route to handler
        try:
            if method == "initialize":
                result = self._handle_initialize(params)
            elif method == "tools/list":
                result = self._handle_list_tools(params)
            elif method == "tools/call":
                result = self._handle_call_tool(params)
            elif method == "ping":
                result = {"pong": True}
            else:
                raise MCPError(MCPErrorCode.METHOD_NOT_FOUND, f"Unknown method: {method}")

            return self._success_response(request_id, result)

        except MCPError as e:
            return self._error_response(request_id, e)
        except Exception as e:
            logger.error(f"Internal error handling {method}: {e}\n{traceback.format_exc()}")
            return self._error_response(
                request_id,
                MCPError(MCPErrorCode.INTERNAL_ERROR, str(e))
            )

    def _success_response(self, request_id: Any, result: Any) -> Dict[str, Any]:
        """Create success response."""
        return {
            "jsonrpc": self.JSONRPC_VERSION,
            "id": request_id,
            "result": result,
        }

    def _error_response(self, request_id: Any, error: MCPError) -> Dict[str, Any]:
        """Create error response."""
        return {
            "jsonrpc": self.JSONRPC_VERSION,
            "id": request_id,
            "error": error.to_dict(),
        }

    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        return {
            "protocolVersion": self.MCP_VERSION,
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
            "capabilities": {
                "tools": {
                    "listChanged": False,  # Tools don't change dynamically
                },
            },
        }

    def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle tools/list request.

        Returns list of available tools with their schemas.
        """
        tools = []
        for tool in self.tools.values():
            definition = tool.get_definition()
            tools.append(definition.to_json_schema())

        return {"tools": tools}

    def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle tools/call request.

        Executes the specified tool with given arguments.
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            raise MCPError(MCPErrorCode.INVALID_PARAMS, "Tool name is required")

        if tool_name not in self.tools:
            raise MCPError(MCPErrorCode.TOOL_NOT_FOUND, f"Tool not found: {tool_name}")

        tool = self.tools[tool_name]

        # Validate arguments
        tool.validate_arguments(arguments)

        # Execute tool
        try:
            result = tool.execute(arguments)
            return result.to_dict()
        except MCPError:
            raise
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}\n{traceback.format_exc()}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Tool execution failed: {e}"
            )


# ============================================================================
# Built-in Tools: FileSystem
# ============================================================================

class FileSystemTool(MCPTool):
    """
    File system operations tool following MCP standard.

    Provides safe read/write operations with path restrictions.

    Operations:
        - read_file: Read file contents
        - write_file: Write content to file
        - list_directory: List directory contents
        - file_info: Get file metadata

    Security:
        - Only allows access to paths within allowed_paths
        - Validates path traversal attacks
        - Configurable write permissions
    """

    def __init__(
        self,
        allowed_paths: Optional[List[str]] = None,
        allow_write: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
    ):
        """
        Initialize FileSystem tool.

        Args:
            allowed_paths: List of allowed path prefixes (None = current directory)
            allow_write: Whether to allow write operations
            max_file_size: Maximum file size for read operations
        """
        self.allowed_paths = [Path(p).resolve() for p in (allowed_paths or ["."])]
        self.allow_write = allow_write
        self.max_file_size = max_file_size

    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def description(self) -> str:
        return "File system operations (read, write, list)"

    def get_parameters(self) -> List[MCPToolParameter]:
        return [
            MCPToolParameter(
                "operation",
                "string",
                "Operation to perform",
                required=True,
                enum=["read_file", "write_file", "list_directory", "file_info"]
            ),
            MCPToolParameter(
                "path",
                "string",
                "File or directory path",
                required=True
            ),
            MCPToolParameter(
                "content",
                "string",
                "Content to write (for write_file)",
                required=False
            ),
            MCPToolParameter(
                "encoding",
                "string",
                "File encoding (default: utf-8)",
                required=False,
                default="utf-8"
            ),
        ]

    def _validate_path(self, path: str) -> Path:
        """
        Validate and resolve path.

        Raises MCPError if path is outside allowed directories.
        """
        try:
            resolved = Path(path).resolve()
        except Exception as e:
            raise MCPError(MCPErrorCode.INVALID_PARAMS, f"Invalid path: {e}")

        # Check if path is within allowed paths
        for allowed in self.allowed_paths:
            try:
                resolved.relative_to(allowed)
                return resolved
            except ValueError:
                continue

        raise MCPError(
            MCPErrorCode.PERMISSION_DENIED,
            f"Path not in allowed directories: {path}"
        )

    def execute(self, arguments: Dict[str, Any]) -> MCPToolResult:
        """Execute file system operation."""
        operation = arguments["operation"]
        path = arguments["path"]
        encoding = arguments.get("encoding", "utf-8")

        if operation == "read_file":
            return self._read_file(path, encoding)
        elif operation == "write_file":
            content = arguments.get("content", "")
            return self._write_file(path, content, encoding)
        elif operation == "list_directory":
            return self._list_directory(path)
        elif operation == "file_info":
            return self._file_info(path)
        else:
            raise MCPError(MCPErrorCode.INVALID_PARAMS, f"Unknown operation: {operation}")

    def _read_file(self, path: str, encoding: str) -> MCPToolResult:
        """Read file contents."""
        resolved = self._validate_path(path)

        if not resolved.exists():
            raise MCPError(MCPErrorCode.RESOURCE_NOT_FOUND, f"File not found: {path}")

        if not resolved.is_file():
            raise MCPError(MCPErrorCode.INVALID_PARAMS, f"Not a file: {path}")

        # Check file size
        size = resolved.stat().st_size
        if size > self.max_file_size:
            raise MCPError(
                MCPErrorCode.INVALID_PARAMS,
                f"File too large: {size} bytes (max: {self.max_file_size})"
            )

        try:
            content = resolved.read_text(encoding=encoding)
            return MCPToolResult.text(content)
        except UnicodeDecodeError:
            # Try reading as binary and return base64
            import base64
            content = base64.b64encode(resolved.read_bytes()).decode('ascii')
            return MCPToolResult(content=[{
                "type": "resource",
                "resource": {
                    "uri": f"file://{resolved}",
                    "mimeType": "application/octet-stream",
                    "blob": content,
                }
            }])

    def _write_file(self, path: str, content: str, encoding: str) -> MCPToolResult:
        """Write content to file."""
        if not self.allow_write:
            raise MCPError(MCPErrorCode.PERMISSION_DENIED, "Write operations not allowed")

        resolved = self._validate_path(path)

        # Create parent directories if needed
        resolved.parent.mkdir(parents=True, exist_ok=True)

        try:
            resolved.write_text(content, encoding=encoding)
            return MCPToolResult.text(f"Successfully wrote {len(content)} characters to {path}")
        except Exception as e:
            raise MCPError(MCPErrorCode.TOOL_EXECUTION_ERROR, f"Write failed: {e}")

    def _list_directory(self, path: str) -> MCPToolResult:
        """List directory contents."""
        resolved = self._validate_path(path)

        if not resolved.exists():
            raise MCPError(MCPErrorCode.RESOURCE_NOT_FOUND, f"Directory not found: {path}")

        if not resolved.is_dir():
            raise MCPError(MCPErrorCode.INVALID_PARAMS, f"Not a directory: {path}")

        entries = []
        for entry in sorted(resolved.iterdir()):
            entry_info = {
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
            }
            if entry.is_file():
                entry_info["size"] = entry.stat().st_size
            entries.append(entry_info)

        return MCPToolResult.json_data({"entries": entries, "count": len(entries)})

    def _file_info(self, path: str) -> MCPToolResult:
        """Get file metadata."""
        resolved = self._validate_path(path)

        if not resolved.exists():
            raise MCPError(MCPErrorCode.RESOURCE_NOT_FOUND, f"Path not found: {path}")

        stat = resolved.stat()

        info = {
            "path": str(resolved),
            "name": resolved.name,
            "type": "directory" if resolved.is_dir() else "file",
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
        }

        if resolved.is_file():
            info["extension"] = resolved.suffix

        return MCPToolResult.json_data(info)


# ============================================================================
# Built-in Tools: Calculator
# ============================================================================

class CalculatorTool(MCPTool):
    """
    Safe mathematical calculator tool.

    Supports basic arithmetic, powers, roots, and common functions.
    Uses Python's ast module for safe expression evaluation.
    """

    ALLOWED_NAMES = {
        'abs', 'round', 'min', 'max', 'sum', 'pow',
        'sqrt', 'sin', 'cos', 'tan', 'log', 'log10', 'exp',
        'pi', 'e', 'inf',
    }

    def __init__(self):
        import math
        self._math_context = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'pi': math.pi,
            'e': math.e,
            'inf': float('inf'),
        }

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate mathematical expressions safely"

    def get_parameters(self) -> List[MCPToolParameter]:
        return [
            MCPToolParameter(
                "expression",
                "string",
                "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')",
                required=True
            ),
        ]

    def execute(self, arguments: Dict[str, Any]) -> MCPToolResult:
        """Evaluate expression safely."""
        import ast

        expression = arguments["expression"]

        try:
            # Parse the expression
            tree = ast.parse(expression, mode='eval')

            # Validate - only allow safe operations
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if node.id not in self.ALLOWED_NAMES and not node.id.startswith('_'):
                        raise MCPError(
                            MCPErrorCode.INVALID_PARAMS,
                            f"Unknown function or variable: {node.id}"
                        )
                elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        if node.func.id not in self.ALLOWED_NAMES:
                            raise MCPError(
                                MCPErrorCode.INVALID_PARAMS,
                                f"Function not allowed: {node.func.id}"
                            )

            # Evaluate
            result = eval(compile(tree, '<expression>', 'eval'), {"__builtins__": {}}, self._math_context)

            return MCPToolResult.json_data({
                "expression": expression,
                "result": result,
            })

        except MCPError:
            raise
        except SyntaxError as e:
            raise MCPError(MCPErrorCode.INVALID_PARAMS, f"Invalid expression syntax: {e}")
        except Exception as e:
            raise MCPError(MCPErrorCode.TOOL_EXECUTION_ERROR, f"Calculation error: {e}")


# ============================================================================
# Built-in Tools: Web Search (Stub)
# ============================================================================

class WebSearchTool(MCPTool):
    """
    Web search tool stub.

    This is a placeholder that demonstrates the interface.
    In production, connect to a search API (Google, Bing, etc.).
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information"

    def get_parameters(self) -> List[MCPToolParameter]:
        return [
            MCPToolParameter(
                "query",
                "string",
                "Search query",
                required=True
            ),
            MCPToolParameter(
                "num_results",
                "number",
                "Number of results to return",
                required=False,
                default=5
            ),
        ]

    def execute(self, arguments: Dict[str, Any]) -> MCPToolResult:
        """Execute search (stub implementation)."""
        query = arguments["query"]
        num_results = arguments.get("num_results", 5)

        # Stub: Return placeholder results
        # In production, call actual search API here
        results = [
            {
                "title": f"Result {i+1} for: {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a placeholder result for the query '{query}'...",
            }
            for i in range(min(num_results, 5))
        ]

        return MCPToolResult.json_data({
            "query": query,
            "results": results,
            "note": "This is a stub implementation. Connect a real search API for production use.",
        })


# ============================================================================
# Tool Registry Helper
# ============================================================================

def create_default_server(
    allowed_fs_paths: Optional[List[str]] = None,
    enable_write: bool = False,
) -> MCPServer:
    """
    Create an MCP server with default tools.

    Args:
        allowed_fs_paths: Allowed paths for filesystem tool
        enable_write: Enable write operations for filesystem

    Returns:
        Configured MCPServer instance
    """
    server = MCPServer(name="vAGI Default MCP Server")

    # Register default tools
    server.register_tool(FileSystemTool(
        allowed_paths=allowed_fs_paths or ["."],
        allow_write=enable_write,
    ))
    server.register_tool(CalculatorTool())
    server.register_tool(WebSearchTool())

    return server


# ============================================================================
# Example Usage
# ============================================================================

def demo_mcp_server():
    """Demonstrate MCP server usage."""
    # Create server with tools
    server = create_default_server(allowed_fs_paths=[".", "/tmp"], enable_write=True)

    print("=== MCP Server Demo ===\n")

    # Initialize
    response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    })
    print("Initialize:", json.dumps(response, indent=2))

    # List tools
    response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    })
    print("\nList Tools:", json.dumps(response, indent=2))

    # Call calculator
    response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "calculator",
            "arguments": {"expression": "sqrt(16) + pow(2, 3)"}
        }
    })
    print("\nCalculator Result:", json.dumps(response, indent=2))

    # List current directory
    response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "filesystem",
            "arguments": {
                "operation": "list_directory",
                "path": "."
            }
        }
    })
    print("\nDirectory Listing:", json.dumps(response, indent=2)[:500] + "...")


if __name__ == "__main__":
    demo_mcp_server()
