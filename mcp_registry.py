# tools/mcp_registry.py
from typing import Dict, Type
from abc import ABC, abstractmethod

class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs):
        pass

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool
        print(f"Tool '{tool.name}' registered.")

    def get_tool(self, tool_name: str) -> BaseTool:
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found.")
        return tool

mcp_registry = ToolRegistry()
