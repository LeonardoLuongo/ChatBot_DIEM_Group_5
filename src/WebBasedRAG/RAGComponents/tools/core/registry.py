from typing import List
from langchain_core.tools import StructuredTool
from .interfaces import ToolProvider

class ToolRegistry:
    """
    Open/Closed Principle (OCP): Questa classe è chiusa alle modifiche ma aperta alle estensioni.
    Per aggiungere nuovi set di tool, basta registrare un nuovo ToolProvider senza toccare questo codice.
    """
    def __init__(self):
        self._providers: List[ToolProvider] = []

    def register_provider(self, provider: ToolProvider) -> None:
        self._providers.append(provider)

    def get_all_tools(self) -> List[StructuredTool]:
        tools = []
        for provider in self._providers:
            tools.extend(provider.get_tools())
        return tools
