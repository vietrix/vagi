"""Tool use and external capabilities for AGI."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ToolRegistry:
    """Registry for available tools."""

    def __init__(self) -> None:
        self.tools: Dict[str, Callable] = {}
        self.tool_descriptions: Dict[str, str] = {}

    def register(
        self,
        name: str,
        function: Callable,
        description: str
    ) -> None:
        """Register a new tool."""
        self.tools[name] = function
        self.tool_descriptions[name] = description

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all available tools."""
        return list(self.tools.keys())

    def get_description(self, name: str) -> Optional[str]:
        """Get tool description."""
        return self.tool_descriptions.get(name)


class ToolSelector(nn.Module):
    """Select appropriate tool for task."""

    def __init__(
        self,
        hidden_size: int,
        num_tools: int,
        context_size: int = 512,
    ) -> None:
        super().__init__()
        self.num_tools = num_tools
        
        self.context_encoder = nn.Sequential(
            nn.Linear(context_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.tool_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_tools)
        )
        
        self.tool_embeddings = nn.Embedding(num_tools, hidden_size)

    def forward(
        self,
        context: torch.Tensor,
        available_tools: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select best tool for context."""
        encoded_context = self.context_encoder(context)
        
        tool_scores = self.tool_scorer(encoded_context)
        
        if available_tools is not None:
            mask = torch.zeros(self.num_tools, device=context.device)
            mask[available_tools] = 1.0
            tool_scores = tool_scores + (1.0 - mask) * -1e9
        
        tool_probs = F.softmax(tool_scores, dim=-1)
        
        selected_tool = torch.argmax(tool_probs, dim=-1)
        
        return selected_tool, tool_probs


class APICallGenerator(nn.Module):
    """Generate API calls from natural language."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_call_length: int = 64,
    ) -> None:
        super().__init__()
        self.max_call_length = max_call_length
        
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(
        self,
        instruction: torch.Tensor,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """Generate API call from instruction."""
        max_length = max_length or self.max_call_length
        
        _, (h_n, c_n) = self.encoder(instruction)
        
        batch_size = instruction.size(0)
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=instruction.device)
        
        generated = []
        hidden = (h_n, c_n)
        
        for _ in range(max_length):
            embedded = self.token_embedding(decoder_input)
            
            output, hidden = self.decoder(embedded, hidden)
            
            logits = self.output_projection(output)
            
            next_token = torch.argmax(logits, dim=-1)
            
            generated.append(next_token)
            decoder_input = next_token
        
        return torch.cat(generated, dim=1)


class ParameterExtractor(nn.Module):
    """Extract parameters from context for tool calls."""

    def __init__(
        self,
        hidden_size: int,
        max_params: int = 10,
    ) -> None:
        super().__init__()
        self.max_params = max_params
        
        self.param_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, max_params),
            nn.Sigmoid()
        )
        
        self.param_extractor = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(max_params)
        ])

    def forward(
        self,
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract parameters from context."""
        param_presence = self.param_detector(context)
        
        extracted_params = []
        for extractor in self.param_extractor:
            param = extractor(context)
            extracted_params.append(param)
        
        params_tensor = torch.stack(extracted_params, dim=1)
        
        masked_params = params_tensor * param_presence.unsqueeze(-1)
        
        return masked_params, param_presence


class ToolExecutor(nn.Module):
    """Execute tools and integrate results."""

    def __init__(
        self,
        hidden_size: int,
        tool_registry: ToolRegistry,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tool_registry = tool_registry
        
        self.result_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.integrator = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute tool with parameters."""
        tool = self.tool_registry.get_tool(tool_name)
        
        if tool is None:
            return {"error": f"Tool {tool_name} not found"}
        
        try:
            result = tool(**parameters)
            return result
        except Exception as e:
            return {"error": str(e)}

    def integrate_result(
        self,
        result: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """Integrate tool result into context."""
        encoded_result = self.result_encoder(result)
        
        combined = torch.cat([context.unsqueeze(1), encoded_result.unsqueeze(1)], dim=1)
        
        _, (h_n, _) = self.integrator(combined)
        
        return h_n.squeeze(0)


class CodeExecutor(nn.Module):
    """Generate and execute code."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_code_length: int = 256,
    ) -> None:
        super().__init__()
        self.max_code_length = max_code_length
        
        self.code_generator = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        self.execution_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def generate_code(
        self,
        specification: torch.Tensor,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """Generate code from specification."""
        max_length = max_length or self.max_code_length
        
        batch_size = specification.size(0)
        
        hidden = None
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=specification.device)
        
        generated_tokens = []
        
        for _ in range(max_length):
            embedded = self.token_embedding(decoder_input)
            
            if hidden is None:
                context = torch.cat([specification.unsqueeze(1), embedded], dim=1)
                output, hidden = self.code_generator(context)
            else:
                output, hidden = self.code_generator(embedded, hidden)
            
            logits = self.output_layer(output[:, -1:, :])
            
            next_token = torch.argmax(logits, dim=-1)
            
            generated_tokens.append(next_token)
            decoder_input = next_token
        
        return torch.cat(generated_tokens, dim=1)

    def predict_execution(
        self,
        code_embedding: torch.Tensor,
        input_state: torch.Tensor
    ) -> torch.Tensor:
        """Predict execution outcome."""
        combined = code_embedding + input_state
        
        predicted_output = self.execution_predictor(combined)
        
        return predicted_output


class ToolUseController(nn.Module):
    """High-level controller for tool use."""

    def __init__(
        self,
        hidden_size: int,
        num_tools: int,
        tool_registry: ToolRegistry,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        
        self.tool_selector = ToolSelector(
            hidden_size=hidden_size,
            num_tools=num_tools
        )
        
        self.param_extractor = ParameterExtractor(
            hidden_size=hidden_size
        )
        
        self.tool_executor = ToolExecutor(
            hidden_size=hidden_size,
            tool_registry=tool_registry
        )
        
        self.should_use_tool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        context: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[bool, Optional[int], Optional[torch.Tensor]]:
        """Decide whether to use tool and which one."""
        use_tool_prob = self.should_use_tool(context)
        
        should_use = use_tool_prob.item() > threshold
        
        if not should_use:
            return False, None, None

        tool_names = self.tool_executor.tool_registry.list_tools()
        if not tool_names:
            return False, None, None

        available_tools = list(range(len(tool_names)))
        selected_tool, tool_probs = self.tool_selector(context, available_tools=available_tools)
        
        params, param_presence = self.param_extractor(context)
        
        return True, selected_tool.item(), params
