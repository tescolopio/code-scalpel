"""
TypeScript/JavaScript AST Analysis Module (Stub).

Code Scalpel Polyglot Architecture - TypeScript Support

This module provides AST parsing and analysis for TypeScript and JavaScript
using tree-sitter for parsing and a normalized IR for cross-language analysis.

Status: STUB (v1.3.0 Target)
Trademark: "Code Scalpel" is a trademark of 3D Tech Solutions LLC.

Architecture:
    TypeScript/JS Source
           ↓
    tree-sitter-typescript (Native Parser)
           ↓
    ESTree-compatible AST
           ↓
    Code Scalpel Normalized IR  ← Same IR as Python
           ↓
    PDG/Symbolic/Security Analysis (Shared)

Dependencies:
    - tree-sitter>=0.21.0
    - tree-sitter-typescript>=0.21.0
    - tree-sitter-javascript>=0.21.0
"""

from .analyzer import TypeScriptAnalyzer
from .parser import TypeScriptParser

__all__ = ["TypeScriptAnalyzer", "TypeScriptParser"]
__version__ = "0.1.0-stub"
