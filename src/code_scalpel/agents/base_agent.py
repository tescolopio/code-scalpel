from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..pdg_tools import ASTAnalyzer, PDGAnalyzer, SymbolicExecutor

class BaseCodeAnalysisAgent(ABC):
    def __init__(self):
        self.ast_analyzer = ASTAnalyzer()
        self.pdg_analyzer = PDGAnalyzer()
        self.symbolic_executor = SymbolicExecutor()
        
    @abstractmethod
    def analyze(self, code: str) -> Dict[str, Any]:
        """Analyze code and return results."""
        pass
    
    @abstractmethod
    def suggest_improvements(self, code: str) -> List[str]:
        """Suggest improvements for the code."""
        pass
    
    def get_analysis_context(self) -> Dict[str, Any]:
        """Get the current analysis context."""
        return {
            "ast_cache_size": len(self.ast_analyzer.ast_cache),
            "pdg_cache_size": len(self.pdg_analyzer.pdg_cache),
            "symbolic_vars": len(self.symbolic_executor.variables)
        }