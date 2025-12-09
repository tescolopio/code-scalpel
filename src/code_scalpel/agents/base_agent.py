from abc import ABC, abstractmethod
from typing import Any

from ..pdg_tools import ASTAnalyzer, PDGAnalyzer
from ..symbolic_execution_tools import SymbolicAnalyzer


class BaseCodeAnalysisAgent(ABC):
    def __init__(self):
        self.ast_analyzer = ASTAnalyzer()
        self.pdg_analyzer = PDGAnalyzer()
        self.symbolic_analyzer = SymbolicAnalyzer()

    @abstractmethod
    def analyze(self, code: str) -> dict[str, Any]:
        """Analyze code and return results."""
        pass

    @abstractmethod
    def suggest_improvements(self, code: str) -> list[str]:
        """Suggest improvements for the code."""
        pass

    def get_analysis_context(self) -> dict[str, Any]:
        """Get the current analysis context."""
        return {
            "ast_cache_size": len(self.ast_analyzer.ast_cache),
            "pdg_cache_size": len(self.pdg_analyzer.pdg_cache),
            "symbolic_vars": len(self.symbolic_executor.variables),
        }
