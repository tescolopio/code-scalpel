from typing import Any

from .base_agent import BaseCodeAnalysisAgent


class CodeReviewAgent(BaseCodeAnalysisAgent):
    def analyze(self, code: str) -> dict[str, Any]:
        """Perform comprehensive code review."""
        ast = self.ast_analyzer.parse_to_ast(code)
        pdg = self.pdg_analyzer.build_pdg(code)

        # Analyze code structure
        functions = self.ast_analyzer.get_function_definitions(ast)
        classes = self.ast_analyzer.get_class_definitions(ast)

        # Analyze dependencies
        deps = self.pdg_analyzer.analyze_dependencies(pdg)

        return {
            "structure": {
                "num_functions": len(functions),
                "num_classes": len(classes),
                "avg_function_complexity": self._calculate_complexity(functions),
            },
            "dependencies": deps,
            "suggestions": self.suggest_improvements(code),
        }

    def suggest_improvements(self, code: str) -> list[str]:
        """Suggest code improvements based on best practices."""
        suggestions = []

        # Analyze code quality
        ast = self.ast_analyzer.parse_to_ast(code)

        # Check function lengths
        for func in self.ast_analyzer.get_function_definitions(ast):
            if len(func.body) > 20:
                suggestions.append(
                    f"Function '{func.name}' is too long ({len(func.body)} lines). "
                    "Consider breaking it into smaller functions."
                )
        return suggestions
