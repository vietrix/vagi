"""I/O components for vAGI including MCP protocol support and model loading."""

from .mcp_interface import (
    MCPServer,
    MCPTool,
    MCPToolResult,
    MCPError,
    MCPErrorCode,
    FileSystemTool,
    CalculatorTool,
    WebSearchTool,
)

from .ui_generator import (
    UIGenerator,
    UIComponent,
    UIComponentType,
    UIAction,
    UIActionType,
)

from .loader import (
    UnifiedModelLoader,
    LoadedModel,
    ModelConfig,
    DynamicInt8Quantizer,
    QuantizationDetector,
    get_recommended_settings,
)

__all__ = [
    # MCP Interface
    "MCPServer",
    "MCPTool",
    "MCPToolResult",
    "MCPError",
    "MCPErrorCode",
    "FileSystemTool",
    "CalculatorTool",
    "WebSearchTool",
    # UI Generator
    "UIGenerator",
    "UIComponent",
    "UIComponentType",
    "UIAction",
    "UIActionType",
    # Model Loader
    "UnifiedModelLoader",
    "LoadedModel",
    "ModelConfig",
    "DynamicInt8Quantizer",
    "QuantizationDetector",
    "get_recommended_settings",
]
