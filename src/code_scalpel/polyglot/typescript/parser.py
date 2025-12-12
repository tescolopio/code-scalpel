"""
TypeScript/JavaScript Parser using tree-sitter.

This parser converts TypeScript/JavaScript source code into an ESTree-compatible
AST structure, then normalizes it to Code Scalpel's internal IR format.

Status: STUB - Core structure only, not yet functional.

Implementation Notes:
    1. tree-sitter provides fast, incremental parsing
    2. ESTree compatibility allows reuse of existing JS tooling patterns
    3. Normalization to shared IR enables cross-language analysis

Integration Difficulty Assessment:
    - Parser: EASY (tree-sitter does the heavy lifting)
    - AST Normalization: MODERATE (ESTree is well-documented)
    - PDG Generation: MODERATE (control flow similar to Python)
    - Symbolic Execution: HARD (JS semantics are complex - coercion, prototypes)
    - Security Analysis: MODERATE (taint tracking transfers cleanly)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TSNodeType(Enum):
    """TypeScript/JavaScript AST node types (ESTree-compatible subset)."""
    
    # Declarations
    FUNCTION_DECLARATION = "function_declaration"
    CLASS_DECLARATION = "class_declaration"
    VARIABLE_DECLARATION = "variable_declaration"
    ARROW_FUNCTION = "arrow_function"
    METHOD_DEFINITION = "method_definition"
    
    # Statements
    IF_STATEMENT = "if_statement"
    FOR_STATEMENT = "for_statement"
    FOR_IN_STATEMENT = "for_in_statement"
    FOR_OF_STATEMENT = "for_of_statement"
    WHILE_STATEMENT = "while_statement"
    RETURN_STATEMENT = "return_statement"
    TRY_STATEMENT = "try_statement"
    THROW_STATEMENT = "throw_statement"
    
    # Expressions
    CALL_EXPRESSION = "call_expression"
    MEMBER_EXPRESSION = "member_expression"
    BINARY_EXPRESSION = "binary_expression"
    ASSIGNMENT_EXPRESSION = "assignment_expression"
    TEMPLATE_LITERAL = "template_literal"
    AWAIT_EXPRESSION = "await_expression"
    
    # TypeScript-specific
    TYPE_ANNOTATION = "type_annotation"
    INTERFACE_DECLARATION = "interface_declaration"
    TYPE_ALIAS_DECLARATION = "type_alias_declaration"
    ENUM_DECLARATION = "enum_declaration"
    
    # Imports/Exports
    IMPORT_STATEMENT = "import_statement"
    EXPORT_STATEMENT = "export_statement"


@dataclass
class TSNode:
    """Represents a node in the TypeScript AST."""
    
    node_type: TSNodeType
    name: str | None = None
    start_line: int = 0
    end_line: int = 0
    start_col: int = 0
    end_col: int = 0
    children: list["TSNode"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # TypeScript-specific
    type_annotation: str | None = None
    is_async: bool = False
    is_exported: bool = False


@dataclass
class TSParseResult:
    """Result of parsing TypeScript/JavaScript code."""
    
    success: bool
    root: TSNode | None = None
    errors: list[str] = field(default_factory=list)
    language: str = "typescript"  # or "javascript"
    
    # Extracted items (for quick access)
    functions: list[dict[str, Any]] = field(default_factory=list)
    classes: list[dict[str, Any]] = field(default_factory=list)
    imports: list[dict[str, Any]] = field(default_factory=list)
    exports: list[dict[str, Any]] = field(default_factory=list)


class TypeScriptParser:
    """
    Parser for TypeScript and JavaScript source code.
    
    Uses tree-sitter for parsing, with fallback to basic regex extraction
    if tree-sitter is not available.
    
    Example:
        >>> parser = TypeScriptParser()
        >>> result = parser.parse('''
        ... function greet(name: string): string {
        ...     return `Hello, ${name}!`;
        ... }
        ... ''')
        >>> print(result.functions[0]['name'])
        'greet'
    """
    
    def __init__(self, language: str = "typescript"):
        """
        Initialize the TypeScript parser.
        
        Args:
            language: "typescript" or "javascript"
        """
        self.language = language
        self._parser = None
        self._tree_sitter_available = False
        self._init_parser()
    
    def _init_parser(self) -> None:
        """Initialize the tree-sitter parser if available."""
        try:
            # Attempt to import tree-sitter
            # This is a STUB - actual implementation would initialize the parser
            # import tree_sitter_typescript as ts_ts
            # import tree_sitter_javascript as ts_js
            # from tree_sitter import Parser
            # 
            # self._parser = Parser()
            # if self.language == "typescript":
            #     self._parser.set_language(ts_ts.language())
            # else:
            #     self._parser.set_language(ts_js.language())
            # self._tree_sitter_available = True
            
            self._tree_sitter_available = False  # STUB
        except ImportError:
            self._tree_sitter_available = False
    
    def parse(self, code: str, filename: str | None = None) -> TSParseResult:
        """
        Parse TypeScript/JavaScript source code.
        
        Args:
            code: Source code to parse
            filename: Optional filename for error reporting
            
        Returns:
            TSParseResult with parsed AST and extracted items
        """
        if not code or not code.strip():
            return TSParseResult(
                success=False,
                errors=["Empty code provided"]
            )
        
        if self._tree_sitter_available:
            return self._parse_with_tree_sitter(code, filename)
        else:
            return self._parse_with_fallback(code, filename)
    
    def _parse_with_tree_sitter(self, code: str, filename: str | None) -> TSParseResult:
        """Parse using tree-sitter (full AST)."""
        # STUB - Would use actual tree-sitter parsing
        raise NotImplementedError("tree-sitter parsing not yet implemented")
    
    def _parse_with_fallback(self, code: str, filename: str | None) -> TSParseResult:
        """
        Fallback parser using regex patterns.
        
        This provides basic structure extraction when tree-sitter is unavailable.
        It won't produce a full AST but can extract top-level declarations.
        """
        import re
        
        functions: list[dict[str, Any]] = []
        classes: list[dict[str, Any]] = []
        imports: list[dict[str, Any]] = []
        exports: list[dict[str, Any]] = []
        
        lines = code.split('\n')
        
        # Function patterns (regular and arrow)
        func_pattern = re.compile(
            r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*'
            r'(?:<[^>]*>)?\s*'  # Optional generics
            r'\(([^)]*)\)'      # Parameters
            r'(?:\s*:\s*([^\{]+))?'  # Optional return type
        )
        
        arrow_pattern = re.compile(
            r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*'
            r'(?::\s*[^=]+)?\s*=\s*'
            r'(?:async\s+)?'
            r'(?:\([^)]*\)|[a-zA-Z_]\w*)\s*=>'
        )
        
        # Class pattern
        class_pattern = re.compile(
            r'^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)'
            r'(?:<[^>]*>)?'  # Optional generics
            r'(?:\s+extends\s+(\w+))?'
            r'(?:\s+implements\s+([^{]+))?'
        )
        
        # Import pattern
        import_pattern = re.compile(
            r'^import\s+(?:{([^}]+)}|(\w+)|\*\s+as\s+(\w+))'
            r'\s+from\s+[\'"]([^\'"]+)[\'"]'
        )
        
        # Export pattern  
        export_pattern = re.compile(
            r'^export\s+(?:default\s+)?(?:const|let|var|function|class|interface|type|enum)\s+(\w+)'
        )
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check function
            match = func_pattern.match(stripped)
            if match:
                functions.append({
                    'name': match.group(1),
                    'params': match.group(2),
                    'return_type': match.group(3).strip() if match.group(3) else None,
                    'line': line_num,
                    'is_async': 'async' in stripped.split('function')[0],
                    'is_exported': stripped.startswith('export')
                })
                continue
            
            # Check arrow function
            match = arrow_pattern.match(stripped)
            if match:
                functions.append({
                    'name': match.group(1),
                    'params': None,  # Would need more parsing
                    'return_type': None,
                    'line': line_num,
                    'is_async': 'async' in stripped,
                    'is_exported': stripped.startswith('export'),
                    'is_arrow': True
                })
                continue
            
            # Check class
            match = class_pattern.match(stripped)
            if match:
                classes.append({
                    'name': match.group(1),
                    'extends': match.group(2),
                    'implements': match.group(3).strip() if match.group(3) else None,
                    'line': line_num,
                    'is_exported': stripped.startswith('export')
                })
                continue
            
            # Check import
            match = import_pattern.match(stripped)
            if match:
                imports.append({
                    'named': match.group(1),
                    'default': match.group(2),
                    'namespace': match.group(3),
                    'source': match.group(4),
                    'line': line_num
                })
                continue
            
            # Check export
            match = export_pattern.match(stripped)
            if match:
                exports.append({
                    'name': match.group(1),
                    'line': line_num,
                    'is_default': 'default' in stripped
                })
        
        return TSParseResult(
            success=True,
            root=None,  # No full AST in fallback mode
            language=self.language,
            functions=functions,
            classes=classes,
            imports=imports,
            exports=exports
        )
    
    def parse_file(self, filepath: str | Path) -> TSParseResult:
        """
        Parse a TypeScript/JavaScript file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            TSParseResult with parsed content
        """
        path = Path(filepath)
        if not path.exists():
            return TSParseResult(
                success=False,
                errors=[f"File not found: {filepath}"]
            )
        
        # Detect language from extension
        if path.suffix in ('.ts', '.tsx', '.mts', '.cts'):
            self.language = "typescript"
        else:
            self.language = "javascript"
        
        code = path.read_text(encoding='utf-8')
        return self.parse(code, str(path))


# Integration difficulty assessment
INTEGRATION_ASSESSMENT = """
TypeScript/JavaScript Integration Difficulty Assessment
========================================================

Component               | Difficulty | LOE (Days) | Notes
------------------------|------------|------------|------
Parser (tree-sitter)    | EASY       | 2-3        | Well-documented, Python bindings exist
AST Normalization       | MODERATE   | 5-7        | ESTree is standard, map to shared IR
Function Extraction     | EASY       | 1-2        | Similar to Python
Class Extraction        | EASY       | 1-2        | Similar to Python
Import Resolution       | MODERATE   | 3-5        | node_modules complexity
Cross-file Analysis     | MODERATE   | 3-5        | Package.json + tsconfig.json parsing
PDG Generation          | MODERATE   | 5-7        | Control flow similar to Python
Taint Analysis          | MODERATE   | 3-5        | Same patterns, different APIs
Symbolic Execution      | HARD       | 10-15      | JS type coercion, prototypes, closures
Test Generation         | HARD       | 5-7        | Jest/Vitest framework support

TOTAL ESTIMATED LOE: 40-58 days for full feature parity

Recommended MVP (v1.3.0):
- Parser + AST Normalization: 7-10 days
- Function/Class Extraction: 2-3 days
- Cross-file Import Resolution: 3-5 days
- Basic Taint Analysis: 3-5 days
= MVP LOE: 15-23 days

Key Technical Challenges:
1. JavaScript's dynamic typing makes symbolic execution complex
2. Prototype chain requires special handling in PDG
3. node_modules resolution follows complex rules
4. JSX/TSX requires additional parser configuration
5. CommonJS vs ESM import differences

Advantages:
1. tree-sitter is battle-tested and fast
2. ESTree is extremely well-documented
3. Large ecosystem of reference implementations
4. TypeScript type annotations help analysis
5. Same MCP interface - no client changes needed
"""

if __name__ == "__main__":
    # Quick test of the stub parser
    parser = TypeScriptParser()
    
    test_code = '''
import { Router, Request, Response } from 'express';
import { UserService } from './services/user';

export class UserController {
    private userService: UserService;
    
    constructor(userService: UserService) {
        this.userService = userService;
    }
    
    async getUser(req: Request, res: Response): Promise<void> {
        const userId = req.params.id;
        const user = await this.userService.findById(userId);
        res.json(user);
    }
}

export async function createRouter(): Promise<Router> {
    const router = Router();
    const controller = new UserController(new UserService());
    
    router.get('/users/:id', (req, res) => controller.getUser(req, res));
    
    return router;
}

const helper = (x: number): number => x * 2;
'''
    
    result = parser.parse(test_code)
    
    print("TypeScript Parser Stub - Test Results")
    print("=" * 50)
    print(f"Success: {result.success}")
    print(f"Language: {result.language}")
    print(f"\nFunctions ({len(result.functions)}):")
    for fn in result.functions:
        print(f"  - {fn['name']} (line {fn['line']}, async={fn.get('is_async', False)})")
    print(f"\nClasses ({len(result.classes)}):")
    for cls in result.classes:
        print(f"  - {cls['name']} (line {cls['line']}, extends={cls.get('extends')})")
    print(f"\nImports ({len(result.imports)}):")
    for imp in result.imports:
        print(f"  - from '{imp['source']}' (line {imp['line']})")
    print(f"\nExports ({len(result.exports)}):")
    for exp in result.exports:
        print(f"  - {exp['name']} (line {exp['line']})")
    
    print("\n" + INTEGRATION_ASSESSMENT)
