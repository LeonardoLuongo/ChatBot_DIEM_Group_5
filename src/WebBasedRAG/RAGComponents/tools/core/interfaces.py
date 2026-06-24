from abc import ABC, abstractmethod
from typing import List
from langchain_core.tools import StructuredTool

class ToolProvider(ABC):
    """
    Interface Segregation Principle (ISP) / Dependency Inversion Principle (DIP):
    Interfaccia base per fornire tool a LangChain. 
    Qualsiasi dominio (WebSearch, EasyCourse, ecc.) deve implementare questo contratto.
    """
    @abstractmethod
    def get_tools(self) -> List[StructuredTool]:
        pass
