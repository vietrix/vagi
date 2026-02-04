#!/usr/bin/env python3
"""
Generative UI Protocol for vAGI.

Implements a prompt-based system for generating structured UI components
in JSON format, enabling the frontend to render pixel-art style dashboards.

When users request data visualization or status information, instead of
plain text descriptions, the model outputs JSON objects that can be
rendered as interactive UI components.

Output Format:
    {
        "type": "pixel-card",
        "data": { ... },
        "style": "retro"
    }

Supported Component Types:
    - pixel-card: Data card with retro pixel art styling
    - status-bar: Progress/status indicator
    - data-table: Tabular data display
    - chart-8bit: 8-bit style chart visualization
    - terminal: Terminal/console output style
    - notification: Alert/notification popup

Usage:
    from core.io.ui_generator import UIGenerator, UIComponentType

    ui_gen = UIGenerator()

    # Add UI generation prompt to model context
    system_prompt = ui_gen.get_system_prompt()

    # Parse model output for UI components
    components = ui_gen.parse_output(model_output)
    for component in components:
        frontend.render(component)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Component Types
# ============================================================================

class UIComponentType(Enum):
    """Supported UI component types for pixel-art dashboard."""
    PIXEL_CARD = "pixel-card"         # Data card with retro styling
    STATUS_BAR = "status-bar"         # Progress/status indicator
    DATA_TABLE = "data-table"         # Tabular data
    CHART_8BIT = "chart-8bit"         # 8-bit style chart
    TERMINAL = "terminal"             # Terminal output
    NOTIFICATION = "notification"     # Alert popup
    STATS_GRID = "stats-grid"         # Grid of statistics
    PROGRESS = "progress"             # Progress bar
    LIST = "list"                     # Bulleted list
    CODE_BLOCK = "code-block"         # Syntax highlighted code


class UIStyle(Enum):
    """UI styling themes."""
    RETRO = "retro"           # Classic pixel art
    NEON = "neon"             # Cyberpunk neon
    TERMINAL = "terminal"     # Green-on-black terminal
    MINIMAL = "minimal"       # Clean minimal design
    VAPORWAVE = "vaporwave"   # 80s/90s aesthetic


# ============================================================================
# Component Data Classes
# ============================================================================

@dataclass
class UIComponent:
    """Base UI component structure."""
    type: str
    data: Dict[str, Any]
    style: str = "retro"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "type": self.type,
            "data": self.data,
            "style": self.style,
            **({"metadata": self.metadata} if self.metadata else {})
        }, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UIComponent":
        """Create from dictionary."""
        return cls(
            type=d.get("type", "pixel-card"),
            data=d.get("data", {}),
            style=d.get("style", "retro"),
            metadata=d.get("metadata", {})
        )


@dataclass
class PixelCard(UIComponent):
    """Pixel-art styled data card."""

    @classmethod
    def create(
        cls,
        title: str,
        value: Union[str, int, float],
        subtitle: Optional[str] = None,
        icon: Optional[str] = None,
        color: str = "blue",
        trend: Optional[str] = None,  # "up", "down", "stable"
    ) -> "PixelCard":
        """Create a pixel card component."""
        return cls(
            type="pixel-card",
            data={
                "title": title,
                "value": value,
                "subtitle": subtitle,
                "icon": icon,
                "color": color,
                "trend": trend,
            },
            style="retro"
        )


@dataclass
class StatusBar(UIComponent):
    """Status/progress bar component."""

    @classmethod
    def create(
        cls,
        label: str,
        value: float,  # 0-100
        max_value: float = 100,
        color: str = "green",
        show_percentage: bool = True,
    ) -> "StatusBar":
        """Create a status bar component."""
        return cls(
            type="status-bar",
            data={
                "label": label,
                "value": value,
                "maxValue": max_value,
                "color": color,
                "showPercentage": show_percentage,
            },
            style="retro"
        )


@dataclass
class DataTable(UIComponent):
    """Tabular data component."""

    @classmethod
    def create(
        cls,
        title: str,
        headers: List[str],
        rows: List[List[Any]],
        sortable: bool = True,
    ) -> "DataTable":
        """Create a data table component."""
        return cls(
            type="data-table",
            data={
                "title": title,
                "headers": headers,
                "rows": rows,
                "sortable": sortable,
            },
            style="retro"
        )


@dataclass
class Chart8Bit(UIComponent):
    """8-bit style chart component."""

    @classmethod
    def create(
        cls,
        title: str,
        chart_type: str,  # "bar", "line", "pie"
        labels: List[str],
        values: List[float],
        colors: Optional[List[str]] = None,
    ) -> "Chart8Bit":
        """Create an 8-bit style chart."""
        return cls(
            type="chart-8bit",
            data={
                "title": title,
                "chartType": chart_type,
                "labels": labels,
                "values": values,
                "colors": colors or ["#00ff00", "#ff00ff", "#00ffff", "#ffff00"],
            },
            style="retro"
        )


@dataclass
class Terminal(UIComponent):
    """Terminal output component."""

    @classmethod
    def create(
        cls,
        title: str = "Terminal",
        lines: Optional[List[str]] = None,
        prompt: str = ">",
    ) -> "Terminal":
        """Create a terminal component."""
        return cls(
            type="terminal",
            data={
                "title": title,
                "lines": lines or [],
                "prompt": prompt,
            },
            style="terminal"
        )


# ============================================================================
# System Prompts
# ============================================================================

UI_GENERATION_SYSTEM_PROMPT = """You are a helpful AI assistant with the ability to generate visual UI components.

## UI Generation Rules

When the user asks for data, status information, statistics, or any information that would benefit from visualization, you MUST output a JSON UI component instead of describing the data in plain text.

### Output Format
```json
{
  "type": "<component-type>",
  "data": { ... },
  "style": "retro"
}
```

### Available Component Types

1. **pixel-card**: For single values/metrics
   ```json
   {
     "type": "pixel-card",
     "data": {
       "title": "Active Users",
       "value": 1234,
       "subtitle": "+12% from last week",
       "icon": "users",
       "color": "blue",
       "trend": "up"
     },
     "style": "retro"
   }
   ```

2. **status-bar**: For progress/completion
   ```json
   {
     "type": "status-bar",
     "data": {
       "label": "Training Progress",
       "value": 75,
       "maxValue": 100,
       "color": "green",
       "showPercentage": true
     },
     "style": "retro"
   }
   ```

3. **data-table**: For tabular data
   ```json
   {
     "type": "data-table",
     "data": {
       "title": "Model Metrics",
       "headers": ["Metric", "Value", "Change"],
       "rows": [
         ["Accuracy", "94.5%", "+2.1%"],
         ["Loss", "0.023", "-0.005"]
       ],
       "sortable": true
     },
     "style": "retro"
   }
   ```

4. **chart-8bit**: For visualizations
   ```json
   {
     "type": "chart-8bit",
     "data": {
       "title": "Weekly Activity",
       "chartType": "bar",
       "labels": ["Mon", "Tue", "Wed", "Thu", "Fri"],
       "values": [12, 19, 8, 15, 22],
       "colors": ["#00ff00"]
     },
     "style": "retro"
   }
   ```

5. **terminal**: For logs/output
   ```json
   {
     "type": "terminal",
     "data": {
       "title": "Build Output",
       "lines": [
         "> npm run build",
         "Building...",
         "✓ Build complete"
       ],
       "prompt": ">"
     },
     "style": "terminal"
   }
   ```

6. **stats-grid**: For multiple statistics
   ```json
   {
     "type": "stats-grid",
     "data": {
       "title": "System Status",
       "items": [
         {"label": "CPU", "value": "45%", "color": "green"},
         {"label": "Memory", "value": "8.2GB", "color": "yellow"},
         {"label": "Disk", "value": "120GB", "color": "blue"}
       ]
     },
     "style": "retro"
   }
   ```

### When to Use UI Components
- User asks: "Show me the status" → Use status-bar or stats-grid
- User asks: "What are the metrics?" → Use data-table or pixel-cards
- User asks: "Display the data" → Use chart-8bit or data-table
- User asks: "Show system info" → Use terminal or stats-grid

### Style Options
- "retro": Classic pixel art (default)
- "neon": Cyberpunk colors
- "terminal": Green-on-black
- "minimal": Clean design

Always output the JSON in a code block with ```json markers so the frontend can parse it."""


UI_RESPONSE_RULES = """
## Important Response Rules

1. If the user asks for data/statistics/status, OUTPUT A JSON UI COMPONENT
2. Wrap JSON in ```json code blocks
3. You can combine text explanation WITH UI components
4. Default style is "retro" (pixel art aesthetic)
5. Use appropriate component types for the data being shown

Example response:
Here's the current training status:

```json
{
  "type": "status-bar",
  "data": {
    "label": "Training Progress",
    "value": 67,
    "maxValue": 100,
    "color": "green"
  },
  "style": "retro"
}
```

The model has completed 67% of training with estimated 2 hours remaining.
"""


# ============================================================================
# UI Generator Class
# ============================================================================

class UIGenerator:
    """
    Generator for structured UI components from model output.

    This class provides:
    1. System prompts to instruct the model to output UI JSON
    2. Parsing logic to extract UI components from model output
    3. Validation of component structure
    4. Helper methods for creating components programmatically
    """

    def __init__(
        self,
        default_style: str = "retro",
        strict_validation: bool = True,
    ):
        """
        Initialize UI generator.

        Args:
            default_style: Default style for components
            strict_validation: Whether to strictly validate component structure
        """
        self.default_style = default_style
        self.strict_validation = strict_validation

        # Valid component types
        self.valid_types = {t.value for t in UIComponentType}

    def get_system_prompt(self, include_rules: bool = True) -> str:
        """
        Get system prompt for UI generation.

        Add this to your model's system prompt to enable
        UI component generation.

        Args:
            include_rules: Include response rules

        Returns:
            System prompt string
        """
        prompt = UI_GENERATION_SYSTEM_PROMPT
        if include_rules:
            prompt += "\n\n" + UI_RESPONSE_RULES
        return prompt

    def parse_output(self, text: str) -> List[UIComponent]:
        """
        Parse model output for UI components.

        Extracts JSON objects from code blocks and validates them
        as UI components.

        Args:
            text: Model output text

        Returns:
            List of UIComponent objects
        """
        components = []

        # Find JSON code blocks
        json_pattern = r'```json\s*\n?(.*?)\n?```'
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            try:
                data = json.loads(match.strip())

                # Handle single component or array
                if isinstance(data, list):
                    for item in data:
                        component = self._validate_and_create(item)
                        if component:
                            components.append(component)
                else:
                    component = self._validate_and_create(data)
                    if component:
                        components.append(component)

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")
                continue

        return components

    def _validate_and_create(self, data: Dict[str, Any]) -> Optional[UIComponent]:
        """Validate and create a UIComponent from dict."""
        if not isinstance(data, dict):
            return None

        # Check required fields
        if "type" not in data:
            if self.strict_validation:
                return None
            data["type"] = "pixel-card"

        if "data" not in data:
            if self.strict_validation:
                return None
            data["data"] = {}

        # Validate type
        if self.strict_validation and data["type"] not in self.valid_types:
            logger.warning(f"Unknown component type: {data['type']}")
            return None

        # Set default style
        if "style" not in data:
            data["style"] = self.default_style

        return UIComponent.from_dict(data)

    def extract_text_and_components(
        self,
        text: str
    ) -> tuple[str, List[UIComponent]]:
        """
        Extract both plain text and UI components from output.

        Args:
            text: Model output

        Returns:
            Tuple of (clean_text, components)
        """
        # Parse components
        components = self.parse_output(text)

        # Remove JSON blocks from text
        clean_text = re.sub(r'```json\s*\n?.*?\n?```', '', text, flags=re.DOTALL)
        clean_text = clean_text.strip()

        return clean_text, components

    # ========================================================================
    # Component Factory Methods
    # ========================================================================

    def create_stats_dashboard(
        self,
        title: str,
        stats: Dict[str, Any],
    ) -> UIComponent:
        """
        Create a stats dashboard from a dictionary.

        Args:
            title: Dashboard title
            stats: Dictionary of stat_name -> value

        Returns:
            UIComponent (stats-grid type)
        """
        items = []
        colors = ["green", "blue", "yellow", "magenta", "cyan"]

        for i, (label, value) in enumerate(stats.items()):
            items.append({
                "label": label,
                "value": str(value),
                "color": colors[i % len(colors)],
            })

        return UIComponent(
            type="stats-grid",
            data={
                "title": title,
                "items": items,
            },
            style=self.default_style
        )

    def create_progress_card(
        self,
        task: str,
        progress: float,
        status: str = "running",
    ) -> UIComponent:
        """Create a progress card for a running task."""
        return UIComponent(
            type="pixel-card",
            data={
                "title": task,
                "value": f"{progress:.1f}%",
                "subtitle": status.capitalize(),
                "icon": "loader" if status == "running" else "check",
                "color": "green" if progress == 100 else "blue",
                "trend": "up" if progress > 0 else "stable",
            },
            style=self.default_style
        )

    def create_error_notification(
        self,
        title: str,
        message: str,
        error_type: Optional[str] = None,
    ) -> UIComponent:
        """Create an error notification component."""
        return UIComponent(
            type="notification",
            data={
                "title": title,
                "message": message,
                "level": "error",
                "errorType": error_type,
                "icon": "alert-triangle",
            },
            style="retro"
        )

    def create_training_dashboard(
        self,
        epoch: int,
        total_epochs: int,
        loss: float,
        accuracy: float,
        learning_rate: float,
        eta: str,
    ) -> List[UIComponent]:
        """
        Create a complete training dashboard with multiple components.

        Returns list of components for a training overview.
        """
        components = []

        # Progress bar
        components.append(UIComponent(
            type="status-bar",
            data={
                "label": f"Training Progress (Epoch {epoch}/{total_epochs})",
                "value": (epoch / total_epochs) * 100,
                "maxValue": 100,
                "color": "green",
                "showPercentage": True,
            },
            style="retro"
        ))

        # Metrics cards
        components.append(UIComponent(
            type="stats-grid",
            data={
                "title": "Training Metrics",
                "items": [
                    {"label": "Loss", "value": f"{loss:.4f}", "color": "yellow"},
                    {"label": "Accuracy", "value": f"{accuracy:.2f}%", "color": "green"},
                    {"label": "Learning Rate", "value": f"{learning_rate:.2e}", "color": "blue"},
                    {"label": "ETA", "value": eta, "color": "cyan"},
                ]
            },
            style="retro"
        ))

        return components


# ============================================================================
# Example Usage
# ============================================================================

def demo_ui_generator():
    """Demonstrate UI generator usage."""
    print("=== UI Generator Demo ===\n")

    ui = UIGenerator()

    # Create sample components
    card = PixelCard.create(
        title="Active Users",
        value=1234,
        subtitle="+12% this week",
        color="green",
        trend="up"
    )
    print("Pixel Card:")
    print(card.to_json())

    print("\n" + "-" * 40 + "\n")

    # Create training dashboard
    dashboard = ui.create_training_dashboard(
        epoch=15,
        total_epochs=50,
        loss=0.0234,
        accuracy=94.5,
        learning_rate=1e-4,
        eta="2h 30m"
    )

    print("Training Dashboard:")
    for comp in dashboard:
        print(comp.to_json())
        print()

    print("-" * 40 + "\n")

    # Parse model output
    sample_output = '''
Here's the current system status:

```json
{
  "type": "stats-grid",
  "data": {
    "title": "System Health",
    "items": [
      {"label": "CPU", "value": "45%", "color": "green"},
      {"label": "Memory", "value": "8.2 GB", "color": "yellow"}
    ]
  },
  "style": "retro"
}
```

Everything is running smoothly!
'''

    text, components = ui.extract_text_and_components(sample_output)
    print(f"Extracted text: {text}")
    print(f"Extracted components: {len(components)}")
    for comp in components:
        print(f"  - {comp.type}: {comp.data.get('title', 'N/A')}")


if __name__ == "__main__":
    demo_ui_generator()
