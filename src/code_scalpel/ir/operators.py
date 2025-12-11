"""
Operator Enums for Unified IR.

These enums normalize operator representations across languages.
The STRUCTURE is normalized here, but SEMANTICS are handled by LanguageSemantics.

Example:
    Python `ast.Add` -> BinaryOperator.ADD
    JS `+` token -> BinaryOperator.ADD
    
But the BEHAVIOR of ADD differs:
    Python: "5" + 3 -> TypeError
    JS: "5" + 3 -> "53"

This difference is handled in semantics.py, not here.
"""

from enum import Enum


class BinaryOperator(Enum):
    """
    Binary operators normalized across languages.

    Maps:
        Python ast.Add, ast.Sub, etc.
        JavaScript +, -, *, /, etc. tokens
    """

    # Arithmetic
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    FLOOR_DIV = "//"  # Python: //, JS: Math.floor(a/b) - semantic difference!
    MOD = "%"
    POW = "**"  # Python: **, JS: **

    # Bitwise
    BIT_AND = "&"
    BIT_OR = "|"
    BIT_XOR = "^"
    LSHIFT = "<<"
    RSHIFT = ">>"

    # Matrix multiplication (Python-only, PEP 465)
    MATMUL = "@"


class CompareOperator(Enum):
    """
    Comparison operators normalized across languages.

    Note: JavaScript has both == (loose) and === (strict).
    Python only has ==. The semantic difference is preserved via source_language.
    """

    # Universal
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="

    # Python-specific
    IS = "is"
    IS_NOT = "is not"
    IN = "in"
    NOT_IN = "not in"

    # JavaScript-specific (strict equality)
    STRICT_EQ = "==="
    STRICT_NE = "!=="


class UnaryOperator(Enum):
    """
    Unary operators normalized across languages.

    Note: Python `not` vs JS `!` have same semantics but different truthiness rules.
    """

    NEG = "-"  # Negation: -x
    POS = "+"  # Unary plus: +x (rarely used, but valid)
    NOT = "not"  # Logical not: Python `not`, JS `!`
    INVERT = "~"  # Bitwise invert: ~x


class BoolOperator(Enum):
    """
    Boolean/logical operators normalized across languages.

    Note: Python `and`/`or` return operand values (short-circuit with value).
    JS `&&`/`||` also short-circuit but coerce to boolean in some contexts.
    """

    AND = "and"  # Python: and, JS: &&
    OR = "or"  # Python: or, JS: ||


class AugAssignOperator(Enum):
    """
    Augmented assignment operators (+=, -=, etc.).

    These map 1:1 to BinaryOperator but are used in different context.
    """

    ADD = "+="
    SUB = "-="
    MUL = "*="
    DIV = "/="
    FLOOR_DIV = "//="
    MOD = "%="
    POW = "**="
    BIT_AND = "&="
    BIT_OR = "|="
    BIT_XOR = "^="
    LSHIFT = "<<="
    RSHIFT = ">>="
    MATMUL = "@="
