"""
JavaScript Normalizer - Convert tree-sitter-javascript CST to Unified IR.

This normalizer handles the NOISE of tree-sitter's Concrete Syntax Tree.
Unlike Python's ast which is already abstract, tree-sitter gives us:

    binary_expression:
        left: number_literal "5"
        "+"  <- Anonymous node (operator)
        right: number_literal "3"

We must extract semantic meaning and discard syntactic tokens.

Supported Constructs:
    - Functions: function declarations, arrow functions, methods
    - Classes: class declarations, methods, constructors
    - Control flow: if/else, for, while, switch
    - Expressions: binary ops, unary ops, calls, member access
    - Modules: import/export (ES6)

CST Node Reference (tree-sitter-javascript):
    https://github.com/tree-sitter/tree-sitter-javascript/blob/master/src/grammar.json
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from .tree_sitter_visitor import TreeSitterVisitor, VisitorContext
from ..nodes import (
    IRModule,
    IRFunctionDef,
    IRClassDef,
    IRIf,
    IRFor,
    IRWhile,
    IRReturn,
    IRAssign,
    IRAugAssign,
    IRExprStmt,
    IRPass,
    IRBreak,
    IRContinue,
    IRBinaryOp,
    IRUnaryOp,
    IRCompare,
    IRBoolOp,
    IRCall,
    IRAttribute,
    IRSubscript,
    IRName,
    IRConstant,
    IRList,
    IRDict,
    IRParameter,
    IRNode,
    IRExpr,
    SourceLocation,
)
from ..operators import (
    BinaryOperator,
    UnaryOperator,
    CompareOperator,
    BoolOperator,
    AugAssignOperator,
)
from .base import BaseNormalizer


# =============================================================================
# Operator Mappings
# =============================================================================

BINARY_OP_MAP = {
    "+": BinaryOperator.ADD,
    "-": BinaryOperator.SUB,
    "*": BinaryOperator.MUL,
    "/": BinaryOperator.DIV,
    "%": BinaryOperator.MOD,
    "**": BinaryOperator.POW,
    "<<": BinaryOperator.LSHIFT,
    ">>": BinaryOperator.RSHIFT,
    ">>>": BinaryOperator.RSHIFT,  # Unsigned right shift -> regular (best effort)
    "&": BinaryOperator.BIT_AND,
    "|": BinaryOperator.BIT_OR,
    "^": BinaryOperator.BIT_XOR,
}

COMPARE_OP_MAP = {
    "==": CompareOperator.EQ,
    "===": CompareOperator.EQ,  # Strict equality -> regular (semantic diff in IR)
    "!=": CompareOperator.NE,
    "!==": CompareOperator.NE,  # Strict inequality
    "<": CompareOperator.LT,
    "<=": CompareOperator.LE,
    ">": CompareOperator.GT,
    ">=": CompareOperator.GE,
    "in": CompareOperator.IN,
    "instanceof": CompareOperator.IN,  # instanceof -> in (best effort)
}

BOOL_OP_MAP = {
    "&&": BoolOperator.AND,
    "||": BoolOperator.OR,
    "??": BoolOperator.OR,  # Nullish coalescing -> OR (best effort)
}

UNARY_OP_MAP = {
    "-": UnaryOperator.NEG,
    "+": UnaryOperator.POS,
    "!": UnaryOperator.NOT,
    "~": UnaryOperator.INVERT,
    "typeof": UnaryOperator.NOT,  # typeof -> NOT (placeholder, needs IR extension)
    "void": UnaryOperator.NOT,    # void -> NOT (placeholder)
}

AUG_ASSIGN_OP_MAP = {
    "+=": AugAssignOperator.ADD,
    "-=": AugAssignOperator.SUB,
    "*=": AugAssignOperator.MUL,
    "/=": AugAssignOperator.DIV,
    "%=": AugAssignOperator.MOD,
    "**=": AugAssignOperator.POW,
    "<<=": AugAssignOperator.LSHIFT,
    ">>=": AugAssignOperator.RSHIFT,
    "&=": AugAssignOperator.BIT_AND,
    "|=": AugAssignOperator.BIT_OR,
    "^=": AugAssignOperator.BIT_XOR,
}


class JavaScriptNormalizer(BaseNormalizer):
    """
    Normalizes JavaScript CST (from tree-sitter) to Unified IR.
    
    This normalizer handles the complexity of tree-sitter's concrete output,
    filtering noise tokens and mapping JavaScript-specific constructs to IR.
    
    Example:
        >>> normalizer = JavaScriptNormalizer()
        >>> ir = normalizer.normalize('''
        ... function add(a, b) {
        ...     return a + b;
        ... }
        ... ''')
        >>> ir.body[0].name
        'add'
    
    Tree-sitter dependency:
        Requires tree_sitter and tree_sitter_javascript packages:
        pip install tree-sitter tree-sitter-javascript
    """
    
    def __init__(self):
        self._filename: str = "<string>"
        self._source: str = ""
        self._parser = None
        self._language = None
        self._ensure_parser()
    
    def _ensure_parser(self) -> None:
        """Lazily initialize tree-sitter parser."""
        if self._parser is not None:
            return
            
        try:
            import tree_sitter_javascript as ts_js
            from tree_sitter import Language, Parser
            
            self._language = Language(ts_js.language())
            self._parser = Parser(self._language)
        except ImportError as e:
            raise ImportError(
                "JavaScriptNormalizer requires tree-sitter packages. "
                "Install with: pip install tree-sitter tree-sitter-javascript"
            ) from e
    
    @property
    def language(self) -> str:
        return "javascript"
    
    def normalize(self, source: str, filename: str = "<string>") -> IRModule:
        """Parse JavaScript source and normalize to IR."""
        self._ensure_parser()
        self._filename = filename
        self._source = source
        
        # Parse with tree-sitter
        tree = self._parser.parse(source.encode("utf-8"))
        root = tree.root_node
        
        # Check for parse errors
        if root.has_error:
            # Find first error node
            error_node = self._find_error_node(root)
            if error_node:
                loc = self._make_loc(error_node)
                raise SyntaxError(f"Parse error at {loc}")
            raise SyntaxError("Parse error in JavaScript source")
        
        # Normalize the program
        return self._normalize_program(root)
    
    def normalize_node(self, node: Any) -> Union[IRNode, List[IRNode], None]:
        """Dispatch to appropriate normalizer based on node type."""
        node_type = node.type
        
        # Map node types to handlers
        method_name = f"_normalize_{node_type}"
        method = getattr(self, method_name, None)
        
        if method is not None:
            return method(node)
        
        # Skip noise nodes silently
        if self._is_noise(node):
            return None
        
        # Warn and skip unknown nodes
        import warnings
        warnings.warn(
            f"JavaScript CST node type '{node_type}' not yet supported. "
            f"At {self._make_loc(node)}"
        )
        return None
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _make_loc(self, node) -> SourceLocation:
        """Create SourceLocation from tree-sitter node."""
        start = node.start_point
        end = node.end_point
        return SourceLocation(
            line=start[0] + 1,  # tree-sitter is 0-indexed
            column=start[1],
            end_line=end[0] + 1,
            end_column=end[1],
            filename=self._filename,
        )
    
    def _get_text(self, node) -> str:
        """Get source text for a node."""
        return self._source[node.start_byte:node.end_byte]
    
    def _is_noise(self, node) -> bool:
        """Check if node is syntactic noise."""
        # Anonymous nodes in tree-sitter are usually punctuation/operators
        if not node.is_named:
            return True
        # Comments
        if node.type in ("comment", "line_comment", "block_comment"):
            return True
        return False
    
    def _find_error_node(self, node):
        """Find first ERROR node in tree."""
        if node.type == "ERROR":
            return node
        for child in node.children:
            error = self._find_error_node(child)
            if error:
                return error
        return None
    
    def _normalize_body(self, nodes) -> List[IRNode]:
        """Normalize a list of statement nodes."""
        result = []
        for node in nodes:
            normalized = self.normalize_node(node)
            if normalized is not None:
                if isinstance(normalized, list):
                    result.extend(normalized)
                else:
                    result.append(normalized)
        return result
    
    def _get_named_children(self, node) -> List:
        """Get only named (non-anonymous) children."""
        return [c for c in node.children if c.is_named]
    
    def _child_by_field(self, node, field: str):
        """Get child by field name, return None if not found."""
        return node.child_by_field_name(field)
    
    # =========================================================================
    # Program / Module
    # =========================================================================
    
    def _normalize_program(self, node) -> IRModule:
        """Normalize the root program node."""
        body = self._normalize_body(self._get_named_children(node))
        
        return self._set_language(IRModule(
            body=body,
            loc=self._make_loc(node),
        ))
    
    # =========================================================================
    # Statements
    # =========================================================================
    
    def _normalize_expression_statement(self, node) -> IRExprStmt:
        """Normalize expression statement (expr;)."""
        expr_node = self._get_named_children(node)[0] if node.children else None
        if expr_node is None:
            return None
        
        expr = self.normalize_node(expr_node)
        if expr is None:
            return None
        
        return self._set_language(IRExprStmt(
            value=expr,
            loc=self._make_loc(node),
        ))
    
    def _normalize_return_statement(self, node) -> IRReturn:
        """Normalize return statement."""
        # return; or return expr;
        # The return value is a named child, not a field
        named_children = self._get_named_children(node)
        value = self.normalize_node(named_children[0]) if named_children else None
        
        return self._set_language(IRReturn(
            value=value,
            loc=self._make_loc(node),
        ))
    
    def _normalize_break_statement(self, node) -> IRBreak:
        """Normalize break statement."""
        return self._set_language(IRBreak(loc=self._make_loc(node)))
    
    def _normalize_continue_statement(self, node) -> IRContinue:
        """Normalize continue statement."""
        return self._set_language(IRContinue(loc=self._make_loc(node)))
    
    def _normalize_empty_statement(self, node) -> None:
        """Empty statement (;) - skip."""
        return None
    
    # =========================================================================
    # Variable Declarations
    # =========================================================================
    
    def _normalize_variable_declaration(self, node) -> Union[IRAssign, List[IRAssign]]:
        """
        Normalize variable declaration (const, let, var).
        
        CST structure:
            variable_declaration:
                "const" | "let" | "var"
                variable_declarator:
                    name: identifier
                    value: expression (optional)
        """
        declarators = [c for c in node.children if c.type == "variable_declarator"]
        
        if len(declarators) == 1:
            return self._normalize_variable_declarator(declarators[0])
        
        # Multiple declarations: const a = 1, b = 2;
        return [self._normalize_variable_declarator(d) for d in declarators]
    
    def _normalize_variable_declarator(self, node) -> IRAssign:
        """Normalize single variable declarator."""
        name_node = self._child_by_field(node, "name")
        value_node = self._child_by_field(node, "value")
        
        target = self.normalize_node(name_node) if name_node else None
        value = self.normalize_node(value_node) if value_node else IRConstant(value=None)
        
        return self._set_language(IRAssign(
            targets=[target],
            value=value,
            loc=self._make_loc(node),
        ))
    
    def _normalize_lexical_declaration(self, node) -> Union[IRAssign, List[IRAssign]]:
        """Normalize lexical declaration (let, const in ES6)."""
        return self._normalize_variable_declaration(node)
    
    # =========================================================================
    # Functions
    # =========================================================================
    
    def _normalize_function_declaration(self, node) -> IRFunctionDef:
        """
        Normalize function declaration.
        
        CST structure:
            function_declaration:
                "async"? (optional)
                "function"
                name: identifier
                parameters: formal_parameters
                body: statement_block
        """
        name_node = self._child_by_field(node, "name")
        params_node = self._child_by_field(node, "parameters")
        body_node = self._child_by_field(node, "body")
        
        name = self._get_text(name_node) if name_node else ""
        params = self._normalize_parameters(params_node) if params_node else []
        body = self._normalize_block(body_node) if body_node else []
        
        # Check for async keyword
        is_async = any(
            c.type == "async" or self._get_text(c) == "async"
            for c in node.children if not c.is_named
        )
        
        return self._set_language(IRFunctionDef(
            name=name,
            params=params,
            body=body,
            is_async=is_async,
            loc=self._make_loc(node),
        ))
    
    def _normalize_arrow_function(self, node) -> IRFunctionDef:
        """
        Normalize arrow function.
        
        CST structure:
            arrow_function:
                "async"? (optional)
                parameter: identifier | formal_parameters
                "=>"
                body: expression | statement_block
        """
        param_node = self._child_by_field(node, "parameter")
        params_node = self._child_by_field(node, "parameters")
        body_node = self._child_by_field(node, "body")
        
        # Parameters can be single identifier or formal_parameters
        if param_node:
            params = [IRParameter(name=self._get_text(param_node))]
        elif params_node:
            params = self._normalize_parameters(params_node)
        else:
            params = []
        
        # Body can be expression or block
        if body_node:
            if body_node.type == "statement_block":
                body = self._normalize_block(body_node)
            else:
                # Expression body: implicit return
                expr = self.normalize_node(body_node)
                body = [IRReturn(value=expr)] if expr else []
        else:
            body = []
        
        is_async = any(
            c.type == "async" or self._get_text(c) == "async"
            for c in node.children if not c.is_named
        )
        
        return self._set_language(IRFunctionDef(
            name="",  # Arrow functions are anonymous
            params=params,
            body=body,
            is_async=is_async,
            loc=self._make_loc(node),
        ))
    
    def _normalize_function_expression(self, node) -> IRFunctionDef:
        """Normalize function expression (anonymous or named)."""
        return self._normalize_function_declaration(node)
    
    def _normalize_generator_function_declaration(self, node) -> IRFunctionDef:
        """Normalize generator function declaration."""
        func = self._normalize_function_declaration(node)
        func.is_generator = True
        return func
    
    def _normalize_parameters(self, node) -> List[IRParameter]:
        """Normalize formal parameters."""
        params = []
        for child in self._get_named_children(node):
            if child.type == "identifier":
                params.append(IRParameter(name=self._get_text(child)))
            elif child.type == "assignment_pattern":
                # Default parameter: a = 1
                name_node = self._child_by_field(child, "left")
                default_node = self._child_by_field(child, "right")
                params.append(IRParameter(
                    name=self._get_text(name_node) if name_node else "",
                    default=self.normalize_node(default_node) if default_node else None,
                ))
            elif child.type == "rest_pattern":
                # Rest parameter: ...args
                name_node = child.children[1] if len(child.children) > 1 else None
                params.append(IRParameter(
                    name=self._get_text(name_node) if name_node else "",
                    is_variadic=True,
                ))
        return params
    
    def _normalize_statement_block(self, node) -> List[IRNode]:
        """Normalize statement block ({ ... })."""
        return self._normalize_block(node)
    
    def _normalize_block(self, node) -> List[IRNode]:
        """Normalize a block of statements."""
        return self._normalize_body(self._get_named_children(node))
    
    # =========================================================================
    # Control Flow
    # =========================================================================
    
    def _normalize_if_statement(self, node) -> IRIf:
        """
        Normalize if statement.
        
        CST structure:
            if_statement:
                "if"
                condition: parenthesized_expression
                consequence: statement
                "else"? (optional)
                alternative: statement (optional)
        """
        cond_node = self._child_by_field(node, "condition")
        cons_node = self._child_by_field(node, "consequence")
        alt_node = self._child_by_field(node, "alternative")
        
        # Condition is wrapped in parentheses, unwrap it
        condition = self._unwrap_expression(cond_node) if cond_node else None
        
        # Consequence
        if cons_node:
            if cons_node.type == "statement_block":
                consequence = self._normalize_block(cons_node)
            else:
                cons = self.normalize_node(cons_node)
                consequence = [cons] if cons else []
        else:
            consequence = []
        
        # Alternative (else branch)
        alternative = []
        if alt_node:
            if alt_node.type == "statement_block":
                alternative = self._normalize_block(alt_node)
            elif alt_node.type == "if_statement":
                # else if -> nested if
                alternative = [self._normalize_if_statement(alt_node)]
            else:
                alt = self.normalize_node(alt_node)
                alternative = [alt] if alt else []
        
        return self._set_language(IRIf(
            test=condition,
            body=consequence,
            orelse=alternative,
            loc=self._make_loc(node),
        ))
    
    def _normalize_while_statement(self, node) -> IRWhile:
        """Normalize while statement."""
        cond_node = self._child_by_field(node, "condition")
        body_node = self._child_by_field(node, "body")
        
        condition = self._unwrap_expression(cond_node) if cond_node else None
        
        if body_node:
            if body_node.type == "statement_block":
                body = self._normalize_block(body_node)
            else:
                stmt = self.normalize_node(body_node)
                body = [stmt] if stmt else []
        else:
            body = []
        
        return self._set_language(IRWhile(
            test=condition,
            body=body,
            loc=self._make_loc(node),
        ))
    
    def _normalize_for_statement(self, node) -> IRFor:
        """
        Normalize for statement.
        
        CST: for (init; condition; update) body
        IR: For loop with init, test, update, body
        """
        init_node = self._child_by_field(node, "initializer")
        cond_node = self._child_by_field(node, "condition")
        update_node = self._child_by_field(node, "update")
        body_node = self._child_by_field(node, "body")
        
        # For traditional for loops, we'll model as IRFor
        # Target is the loop variable (from init)
        # Iter is a synthetic range (or update expression)
        
        init = self.normalize_node(init_node) if init_node else None
        condition = self.normalize_node(cond_node) if cond_node else None
        update = self.normalize_node(update_node) if update_node else None
        
        if body_node:
            if body_node.type == "statement_block":
                body = self._normalize_block(body_node)
            else:
                stmt = self.normalize_node(body_node)
                body = [stmt] if stmt else []
        else:
            body = []
        
        # Traditional for loops don't map cleanly to IR for-each
        # We'll use IRWhile as the best approximation
        # TODO: Add IRForLoop to IR for C-style for loops
        
        # For now, return as IRWhile with init prepended
        while_node = IRWhile(
            test=condition if condition else IRConstant(value=True),
            body=body + ([IRExprStmt(value=update)] if update else []),
            loc=self._make_loc(node),
        )
        
        if init:
            # Return list: init statement + while loop
            if isinstance(init, list):
                return init + [self._set_language(while_node)]
            return [init, self._set_language(while_node)]
        
        return self._set_language(while_node)
    
    def _normalize_for_in_statement(self, node) -> IRFor:
        """Normalize for-in statement (for (x in obj))."""
        left_node = self._child_by_field(node, "left")
        right_node = self._child_by_field(node, "right")
        body_node = self._child_by_field(node, "body")
        
        # Left can be variable declaration or identifier
        if left_node:
            if left_node.type in ("variable_declaration", "lexical_declaration"):
                declarator = [c for c in left_node.children if c.type == "variable_declarator"][0]
                name_node = self._child_by_field(declarator, "name")
                target = IRName(name=self._get_text(name_node)) if name_node else None
            else:
                target = self.normalize_node(left_node)
        else:
            target = None
        
        iter_expr = self.normalize_node(right_node) if right_node else None
        
        if body_node:
            if body_node.type == "statement_block":
                body = self._normalize_block(body_node)
            else:
                stmt = self.normalize_node(body_node)
                body = [stmt] if stmt else []
        else:
            body = []
        
        return self._set_language(IRFor(
            target=target,
            iter=iter_expr,
            body=body,
            loc=self._make_loc(node),
        ))
    
    def _normalize_for_of_statement(self, node) -> IRFor:
        """Normalize for-of statement (for (x of iterable))."""
        # Same structure as for-in
        return self._normalize_for_in_statement(node)
    
    # =========================================================================
    # Expressions
    # =========================================================================
    
    def _unwrap_expression(self, node) -> IRExpr:
        """Unwrap parenthesized expression if needed."""
        if node.type == "parenthesized_expression":
            inner = self._get_named_children(node)
            if inner:
                return self.normalize_node(inner[0])
        return self.normalize_node(node)
    
    def _normalize_parenthesized_expression(self, node) -> IRExpr:
        """Normalize parenthesized expression."""
        inner = self._get_named_children(node)
        if inner:
            return self.normalize_node(inner[0])
        return None
    
    def _normalize_binary_expression(self, node) -> Union[IRBinaryOp, IRCompare, IRBoolOp]:
        """
        Normalize binary expression.
        
        CST structure:
            binary_expression:
                left: expression
                operator (anonymous)
                right: expression
        """
        left_node = self._child_by_field(node, "left")
        right_node = self._child_by_field(node, "right")
        
        # Find operator (anonymous node between left and right)
        op_text = None
        for child in node.children:
            if not child.is_named:
                text = self._get_text(child).strip()
                if text:
                    op_text = text
                    break
        
        left = self.normalize_node(left_node) if left_node else None
        right = self.normalize_node(right_node) if right_node else None
        loc = self._make_loc(node)
        
        # Determine operator type
        if op_text in BINARY_OP_MAP:
            return self._set_language(IRBinaryOp(
                op=BINARY_OP_MAP[op_text],
                left=left,
                right=right,
                loc=loc,
            ))
        elif op_text in COMPARE_OP_MAP:
            return self._set_language(IRCompare(
                ops=[COMPARE_OP_MAP[op_text]],
                left=left,
                comparators=[right],
                loc=loc,
            ))
        elif op_text in BOOL_OP_MAP:
            return self._set_language(IRBoolOp(
                op=BOOL_OP_MAP[op_text],
                values=[left, right],
                loc=loc,
            ))
        else:
            import warnings
            warnings.warn(f"Unknown binary operator '{op_text}' at {loc}")
            # Return as binary op with ADD as placeholder
            return self._set_language(IRBinaryOp(
                op=BinaryOperator.ADD,
                left=left,
                right=right,
                loc=loc,
            ))
    
    def _normalize_unary_expression(self, node) -> IRUnaryOp:
        """Normalize unary expression."""
        arg_node = self._child_by_field(node, "argument")
        
        # Find operator
        op_text = None
        for child in node.children:
            if not child.is_named:
                text = self._get_text(child).strip()
                if text and text != "(":
                    op_text = text
                    break
        
        operand = self.normalize_node(arg_node) if arg_node else None
        op = UNARY_OP_MAP.get(op_text, UnaryOperator.NEG)
        
        return self._set_language(IRUnaryOp(
            op=op,
            operand=operand,
            loc=self._make_loc(node),
        ))
    
    def _normalize_update_expression(self, node) -> IRAugAssign:
        """Normalize update expression (++x, x++, --x, x--)."""
        arg_node = self._child_by_field(node, "argument")
        
        # Find operator
        op_text = None
        for child in node.children:
            if not child.is_named:
                text = self._get_text(child).strip()
                if text in ("++", "--"):
                    op_text = text
                    break
        
        target = self.normalize_node(arg_node) if arg_node else None
        op = AugAssignOperator.ADD if op_text == "++" else AugAssignOperator.SUB
        
        return self._set_language(IRAugAssign(
            target=target,
            op=op,
            value=IRConstant(value=1),
            loc=self._make_loc(node),
        ))
    
    def _normalize_assignment_expression(self, node) -> IRAssign:
        """Normalize assignment expression."""
        left_node = self._child_by_field(node, "left")
        right_node = self._child_by_field(node, "right")
        
        target = self.normalize_node(left_node) if left_node else None
        value = self.normalize_node(right_node) if right_node else None
        
        return self._set_language(IRAssign(
            targets=[target],
            value=value,
            loc=self._make_loc(node),
        ))
    
    def _normalize_augmented_assignment_expression(self, node) -> IRAugAssign:
        """Normalize augmented assignment (+=, -=, etc.)."""
        left_node = self._child_by_field(node, "left")
        right_node = self._child_by_field(node, "right")
        
        # Find operator
        op_text = None
        for child in node.children:
            if not child.is_named:
                text = self._get_text(child).strip()
                if text in AUG_ASSIGN_OP_MAP:
                    op_text = text
                    break
        
        target = self.normalize_node(left_node) if left_node else None
        value = self.normalize_node(right_node) if right_node else None
        op = AUG_ASSIGN_OP_MAP.get(op_text, AugAssignOperator.ADD)
        
        return self._set_language(IRAugAssign(
            target=target,
            op=op,
            value=value,
            loc=self._make_loc(node),
        ))
    
    def _normalize_call_expression(self, node) -> IRCall:
        """Normalize function call."""
        func_node = self._child_by_field(node, "function")
        args_node = self._child_by_field(node, "arguments")
        
        func = self.normalize_node(func_node) if func_node else None
        
        args = []
        if args_node:
            for arg in self._get_named_children(args_node):
                arg_ir = self.normalize_node(arg)
                if arg_ir:
                    args.append(arg_ir)
        
        return self._set_language(IRCall(
            func=func,
            args=args,
            loc=self._make_loc(node),
        ))
    
    def _normalize_member_expression(self, node) -> Union[IRAttribute, IRSubscript]:
        """Normalize member access (obj.prop or obj[key])."""
        obj_node = self._child_by_field(node, "object")
        prop_node = self._child_by_field(node, "property")
        
        obj = self.normalize_node(obj_node) if obj_node else None
        
        # Check if bracket notation (computed)
        is_computed = any(
            self._get_text(c) == "[" for c in node.children if not c.is_named
        )
        
        if is_computed and prop_node:
            # obj[key] -> IRSubscript
            key = self.normalize_node(prop_node)
            return self._set_language(IRSubscript(
                value=obj,
                slice=key,
                loc=self._make_loc(node),
            ))
        else:
            # obj.prop -> IRAttribute
            attr = self._get_text(prop_node) if prop_node else ""
            return self._set_language(IRAttribute(
                value=obj,
                attr=attr,
                loc=self._make_loc(node),
            ))
    
    def _normalize_subscript_expression(self, node) -> IRSubscript:
        """Normalize subscript access (obj[key])."""
        obj_node = self._child_by_field(node, "object")
        index_node = self._child_by_field(node, "index")
        
        obj = self.normalize_node(obj_node) if obj_node else None
        index = self.normalize_node(index_node) if index_node else None
        
        return self._set_language(IRSubscript(
            value=obj,
            slice=index,
            loc=self._make_loc(node),
        ))
    
    # =========================================================================
    # Literals / Atoms
    # =========================================================================
    
    def _normalize_identifier(self, node) -> IRName:
        """Normalize identifier."""
        return self._set_language(IRName(
            id=self._get_text(node),
            loc=self._make_loc(node),
        ))
    
    def _normalize_property_identifier(self, node) -> IRName:
        """Normalize property identifier (in member expressions)."""
        return self._normalize_identifier(node)
    
    def _normalize_number(self, node) -> IRConstant:
        """Normalize number literal."""
        text = self._get_text(node)
        try:
            if "." in text or "e" in text.lower():
                value = float(text)
            else:
                value = int(text, 0)  # Auto-detect base (0x, 0o, 0b)
        except ValueError:
            value = float(text)
        
        return self._set_language(IRConstant(
            value=value,
            loc=self._make_loc(node),
        ))
    
    def _normalize_string(self, node) -> IRConstant:
        """Normalize string literal."""
        text = self._get_text(node)
        # Remove quotes
        if text.startswith(('"""', "'''")):
            value = text[3:-3]
        elif text.startswith(('"', "'")):
            value = text[1:-1]
        else:
            value = text
        
        return self._set_language(IRConstant(
            value=value,
            loc=self._make_loc(node),
        ))
    
    def _normalize_template_string(self, node) -> IRConstant:
        """Normalize template string (`...`)."""
        text = self._get_text(node)
        # For now, treat as plain string
        value = text[1:-1] if text.startswith("`") else text
        
        return self._set_language(IRConstant(
            value=value,
            loc=self._make_loc(node),
        ))
    
    def _normalize_true(self, node) -> IRConstant:
        """Normalize true literal."""
        return self._set_language(IRConstant(value=True, loc=self._make_loc(node)))
    
    def _normalize_false(self, node) -> IRConstant:
        """Normalize false literal."""
        return self._set_language(IRConstant(value=False, loc=self._make_loc(node)))
    
    def _normalize_null(self, node) -> IRConstant:
        """Normalize null literal."""
        return self._set_language(IRConstant(value=None, loc=self._make_loc(node)))
    
    def _normalize_undefined(self, node) -> IRConstant:
        """Normalize undefined literal."""
        return self._set_language(IRConstant(value=None, loc=self._make_loc(node)))
    
    def _normalize_array(self, node) -> IRList:
        """Normalize array literal."""
        elements = []
        for child in self._get_named_children(node):
            elem = self.normalize_node(child)
            if elem:
                elements.append(elem)
        
        return self._set_language(IRList(
            elts=elements,
            loc=self._make_loc(node),
        ))
    
    def _normalize_object(self, node) -> IRDict:
        """Normalize object literal."""
        keys = []
        values = []
        
        for child in self._get_named_children(node):
            if child.type == "pair":
                key_node = self._child_by_field(child, "key")
                value_node = self._child_by_field(child, "value")
                
                if key_node:
                    key_text = self._get_text(key_node)
                    # Key can be identifier or string
                    if key_node.type == "string":
                        key_text = key_text[1:-1]  # Remove quotes
                    keys.append(IRConstant(value=key_text))
                
                if value_node:
                    values.append(self.normalize_node(value_node))
            elif child.type == "shorthand_property_identifier":
                # { foo } -> { foo: foo }
                name = self._get_text(child)
                keys.append(IRConstant(value=name))
                values.append(IRName(name=name))
        
        return self._set_language(IRDict(
            keys=keys,
            values=values,
            loc=self._make_loc(node),
        ))
    
    # =========================================================================
    # Classes
    # =========================================================================
    
    def _normalize_class_declaration(self, node) -> IRClassDef:
        """Normalize class declaration."""
        name_node = self._child_by_field(node, "name")
        heritage_node = self._child_by_field(node, "heritage")  # extends
        body_node = self._child_by_field(node, "body")
        
        name = self._get_text(name_node) if name_node else ""
        
        # Extract base class from heritage clause
        bases = []
        if heritage_node:
            for child in self._get_named_children(heritage_node):
                base = self.normalize_node(child)
                if base:
                    bases.append(base)
        
        # Normalize class body
        body = []
        if body_node:
            for member in self._get_named_children(body_node):
                normalized = self.normalize_node(member)
                if normalized:
                    if isinstance(normalized, list):
                        body.extend(normalized)
                    else:
                        body.append(normalized)
        
        return self._set_language(IRClassDef(
            name=name,
            bases=bases,
            body=body,
            loc=self._make_loc(node),
        ))
    
    def _normalize_class_body(self, node) -> List[IRNode]:
        """Normalize class body."""
        return self._normalize_body(self._get_named_children(node))
    
    def _normalize_method_definition(self, node) -> IRFunctionDef:
        """Normalize class method definition."""
        name_node = self._child_by_field(node, "name")
        params_node = self._child_by_field(node, "parameters")
        body_node = self._child_by_field(node, "body")
        
        name = self._get_text(name_node) if name_node else ""
        params = self._normalize_parameters(params_node) if params_node else []
        body = self._normalize_block(body_node) if body_node else []
        
        # Check for async, static, get, set keywords
        is_async = any(
            self._get_text(c) == "async" for c in node.children if not c.is_named
        )
        
        return self._set_language(IRFunctionDef(
            name=name,
            params=params,
            body=body,
            is_async=is_async,
            loc=self._make_loc(node),
        ))
    
    # =========================================================================
    # Try/Catch/Finally (Stub - IRTry not yet in IR)
    # =========================================================================
    
    def _normalize_try_statement(self, node) -> None:
        """
        Normalize try statement.
        
        NOTE: IRTry/IRRaise not yet implemented in IR.
        For now, we skip try/catch blocks and return None.
        TODO: Add IRTry, IRRaise to ir/nodes.py
        """
        import warnings
        warnings.warn(
            f"try/catch not yet supported in IR. Skipping at {self._make_loc(node)}"
        )
        return None
    
    def _normalize_throw_statement(self, node) -> None:
        """
        Normalize throw statement.
        
        NOTE: IRRaise not yet implemented in IR.
        TODO: Add IRRaise to ir/nodes.py
        """
        import warnings
        warnings.warn(
            f"throw not yet supported in IR. Skipping at {self._make_loc(node)}"
        )
        return None
