"""
Language Semantics - Behavioral Dispatch for Cross-Language Analysis.

This module separates STRUCTURE (IR nodes) from BEHAVIOR (language semantics).

The Problem:
    IRBinaryOp(op=ADD) looks the same for Python and JavaScript.
    But "5" + 3 behaves differently:
        - Python: TypeError (cannot concatenate str and int)
        - JavaScript: "53" (implicit coercion)

The Solution:
    The interpreter walks the IR (structure) but delegates operations
    to a LanguageSemantics instance (behavior).

Example:
    >>> semantics = JavaScriptSemantics()
    >>> semantics.binary_add("5", 3)
    "53"
    
    >>> semantics = PythonSemantics()
    >>> semantics.binary_add("5", 3)
    TypeError: can only concatenate str (not "int") to str
"""

from abc import ABC, abstractmethod
from typing import Any, Union

from z3 import (
    ExprRef,
    ArithRef,
    BoolRef,
    SeqRef,
    IntVal,
    BoolVal,
    StringVal,
    Concat,
    And,
    Or,
    Not,
    If,
)


class LanguageSemantics(ABC):
    """
    Abstract base class for language-specific operation semantics.
    
    Subclasses implement the BEHAVIOR of operators for their language.
    The interpreter calls these methods when evaluating IR nodes.
    
    All methods accept Z3 symbolic values and return Z3 symbolic values.
    This allows symbolic execution across languages.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the language name."""
        pass
    
    # =========================================================================
    # Binary Arithmetic Operations
    # =========================================================================
    
    @abstractmethod
    def binary_add(self, left: Any, right: Any) -> Any:
        """
        Implement the + operator.
        
        Semantic differences:
            Python: str + int -> TypeError
            JavaScript: str + int -> string concatenation
        """
        pass
    
    @abstractmethod
    def binary_sub(self, left: Any, right: Any) -> Any:
        """Implement the - operator."""
        pass
    
    @abstractmethod
    def binary_mul(self, left: Any, right: Any) -> Any:
        """
        Implement the * operator.
        
        Semantic differences:
            Python: "ab" * 3 -> "ababab"
            JavaScript: "ab" * 3 -> NaN
        """
        pass
    
    @abstractmethod
    def binary_div(self, left: Any, right: Any) -> Any:
        """
        Implement the / operator.
        
        Semantic differences:
            Python 3: 5 / 2 -> 2.5 (true division)
            JavaScript: 5 / 2 -> 2.5
        """
        pass
    
    @abstractmethod
    def binary_floor_div(self, left: Any, right: Any) -> Any:
        """
        Implement floor division.
        
        Python: 5 // 2 -> 2
        JavaScript: Math.floor(5 / 2) -> 2 (no // operator)
        """
        pass
    
    @abstractmethod
    def binary_mod(self, left: Any, right: Any) -> Any:
        """
        Implement the % operator.
        
        Semantic differences:
            Python: -7 % 3 -> 2 (result has sign of divisor)
            JavaScript: -7 % 3 -> -1 (result has sign of dividend)
        """
        pass
    
    @abstractmethod
    def binary_pow(self, left: Any, right: Any) -> Any:
        """Implement the ** operator."""
        pass
    
    # =========================================================================
    # Comparison Operations
    # =========================================================================
    
    @abstractmethod
    def compare_eq(self, left: Any, right: Any) -> Any:
        """
        Implement == comparison.
        
        Semantic differences:
            Python: 1 == "1" -> False
            JavaScript: 1 == "1" -> True (type coercion)
        """
        pass
    
    @abstractmethod
    def compare_strict_eq(self, left: Any, right: Any) -> Any:
        """
        Implement === comparison (JS only).
        
        JavaScript: 1 === "1" -> False (no type coercion)
        Python: N/A (always strict)
        """
        pass
    
    @abstractmethod
    def compare_lt(self, left: Any, right: Any) -> Any:
        """Implement < comparison."""
        pass
    
    @abstractmethod
    def compare_le(self, left: Any, right: Any) -> Any:
        """Implement <= comparison."""
        pass
    
    @abstractmethod
    def compare_gt(self, left: Any, right: Any) -> Any:
        """Implement > comparison."""
        pass
    
    @abstractmethod
    def compare_ge(self, left: Any, right: Any) -> Any:
        """Implement >= comparison."""
        pass
    
    # =========================================================================
    # Boolean Operations
    # =========================================================================
    
    @abstractmethod
    def bool_and(self, left: Any, right: Any) -> Any:
        """
        Implement logical AND.
        
        Both Python and JS short-circuit, but return values differ:
            Python: 0 and "hello" -> 0 (returns first falsy)
            JavaScript: 0 && "hello" -> 0 (same behavior)
        """
        pass
    
    @abstractmethod
    def bool_or(self, left: Any, right: Any) -> Any:
        """
        Implement logical OR.
        
        Both Python and JS short-circuit:
            Python: 0 or "hello" -> "hello"
            JavaScript: 0 || "hello" -> "hello"
        """
        pass
    
    @abstractmethod
    def bool_not(self, operand: Any) -> Any:
        """
        Implement logical NOT.
        
        Truthiness rules differ:
            Python: not [] -> True (empty list is falsy)
            JavaScript: ![] -> False (empty array is truthy!)
        """
        pass
    
    # =========================================================================
    # Type Coercion
    # =========================================================================
    
    @abstractmethod
    def to_boolean(self, value: Any) -> Any:
        """
        Convert value to boolean.
        
        Truthiness differs significantly:
            Python falsy: None, False, 0, "", [], {}, set()
            JavaScript falsy: null, undefined, false, 0, "", NaN
            JavaScript truthy: [], {} (empty array/object are truthy!)
        """
        pass
    
    @abstractmethod
    def to_string(self, value: Any) -> Any:
        """Convert value to string."""
        pass
    
    @abstractmethod
    def to_number(self, value: Any) -> Any:
        """
        Convert value to number.
        
        JavaScript: Number("42") -> 42, Number("hello") -> NaN
        Python: int("42") -> 42, int("hello") -> ValueError
        """
        pass


class PythonSemantics(LanguageSemantics):
    """
    Python language semantics.
    
    Key characteristics:
        - Strong typing (no implicit coercion in +)
        - Truthy: non-empty sequences, non-zero numbers
        - Modulo has sign of divisor
    """
    
    @property
    def name(self) -> str:
        return "python"
    
    # =========================================================================
    # Binary Arithmetic - Python is strongly typed
    # =========================================================================
    
    def binary_add(self, left: Any, right: Any) -> Any:
        """
        Python +: No implicit type coercion.
        
        str + str -> str (concatenation)
        int + int -> int
        str + int -> TypeError
        """
        # For symbolic execution, we check types
        if isinstance(left, SeqRef) and isinstance(right, SeqRef):
            # String concatenation
            return Concat(left, right)
        elif isinstance(left, ArithRef) and isinstance(right, ArithRef):
            # Numeric addition
            return left + right
        elif isinstance(left, SeqRef) or isinstance(right, SeqRef):
            # Mixed string and non-string: Python raises TypeError
            raise TypeError(
                f"can only concatenate str (not \"{type(right).__name__}\") to str"
            )
        else:
            # Concrete values - use Python's native behavior
            return left + right
    
    def binary_sub(self, left: Any, right: Any) -> Any:
        """Python -: Numeric subtraction only."""
        if isinstance(left, (ArithRef, int, float)) and isinstance(right, (ArithRef, int, float)):
            return left - right
        raise TypeError(f"unsupported operand type(s) for -")
    
    def binary_mul(self, left: Any, right: Any) -> Any:
        """
        Python *: Numeric multiplication OR string repetition.
        
        "ab" * 3 -> "ababab"
        3 * "ab" -> "ababab"
        """
        # String repetition is complex in Z3, skip for now
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left * right
        return left * right
    
    def binary_div(self, left: Any, right: Any) -> Any:
        """Python /: True division (always returns float in Python 3)."""
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            # Z3 integer division - note: this is floor division
            # For true division we'd need Real sort
            return left / right
        return left / right
    
    def binary_floor_div(self, left: Any, right: Any) -> Any:
        """Python //: Floor division."""
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left / right  # Z3 Int division is floor division
        return left // right
    
    def binary_mod(self, left: Any, right: Any) -> Any:
        """
        Python %: Modulo with sign of divisor.
        
        -7 % 3 -> 2 (not -1)
        """
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left % right
        return left % right
    
    def binary_pow(self, left: Any, right: Any) -> Any:
        """Python **: Exponentiation."""
        # Z3 doesn't have native power, would need unrolling or approximation
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left ** right
        raise NotImplementedError("Symbolic exponentiation not supported")
    
    # =========================================================================
    # Comparison - Python is always strict
    # =========================================================================
    
    def compare_eq(self, left: Any, right: Any) -> Any:
        """Python ==: Strict equality (no type coercion)."""
        if isinstance(left, ExprRef) or isinstance(right, ExprRef):
            return left == right
        return left == right
    
    def compare_strict_eq(self, left: Any, right: Any) -> Any:
        """Python doesn't have ===, delegate to ==."""
        return self.compare_eq(left, right)
    
    def compare_lt(self, left: Any, right: Any) -> Any:
        """Python <: Less than."""
        return left < right
    
    def compare_le(self, left: Any, right: Any) -> Any:
        """Python <=: Less than or equal."""
        return left <= right
    
    def compare_gt(self, left: Any, right: Any) -> Any:
        """Python >: Greater than."""
        return left > right
    
    def compare_ge(self, left: Any, right: Any) -> Any:
        """Python >=: Greater than or equal."""
        return left >= right
    
    # =========================================================================
    # Boolean Operations
    # =========================================================================
    
    def bool_and(self, left: Any, right: Any) -> Any:
        """Python and: Short-circuit, returns operand value."""
        if isinstance(left, BoolRef) and isinstance(right, BoolRef):
            return And(left, right)
        # Concrete: return first falsy or last truthy
        if not left:
            return left
        return right
    
    def bool_or(self, left: Any, right: Any) -> Any:
        """Python or: Short-circuit, returns operand value."""
        if isinstance(left, BoolRef) and isinstance(right, BoolRef):
            return Or(left, right)
        # Concrete: return first truthy or last falsy
        if left:
            return left
        return right
    
    def bool_not(self, operand: Any) -> Any:
        """Python not: Logical negation."""
        if isinstance(operand, BoolRef):
            return Not(operand)
        return not operand
    
    # =========================================================================
    # Type Coercion
    # =========================================================================
    
    def to_boolean(self, value: Any) -> Any:
        """
        Python truthiness:
            Falsy: None, False, 0, 0.0, "", [], {}, set()
            Everything else is truthy.
        """
        if isinstance(value, BoolRef):
            return value
        return bool(value)
    
    def to_string(self, value: Any) -> Any:
        """Python str(): Convert to string."""
        if isinstance(value, SeqRef):
            return value
        return str(value)
    
    def to_number(self, value: Any) -> Any:
        """
        Python int()/float(): Strict conversion.
        
        Raises ValueError on invalid input.
        """
        if isinstance(value, ArithRef):
            return value
        if isinstance(value, str):
            return int(value)  # May raise ValueError
        return int(value)


class JavaScriptSemantics(LanguageSemantics):
    """
    JavaScript language semantics.
    
    Key characteristics:
        - Weak typing (implicit coercion everywhere)
        - Truthy: everything except null, undefined, false, 0, "", NaN
        - Empty array [] is TRUTHY (unlike Python!)
        - Modulo has sign of dividend
    """
    
    @property
    def name(self) -> str:
        return "javascript"
    
    # =========================================================================
    # Binary Arithmetic - JavaScript coerces types
    # =========================================================================
    
    def binary_add(self, left: Any, right: Any) -> Any:
        """
        JavaScript +: String wins!
        
        "5" + 3 -> "53"
        5 + "3" -> "53"
        5 + 3 -> 8
        """
        # Check if either operand is a string
        left_is_string = isinstance(left, (str, SeqRef))
        right_is_string = isinstance(right, (str, SeqRef))
        
        if left_is_string or right_is_string:
            # String concatenation with coercion
            if isinstance(left, SeqRef) and isinstance(right, SeqRef):
                return Concat(left, right)
            # Concrete: coerce to string and concatenate
            return str(left) + str(right)
        else:
            # Numeric addition
            if isinstance(left, ArithRef) and isinstance(right, ArithRef):
                return left + right
            return left + right
    
    def binary_sub(self, left: Any, right: Any) -> Any:
        """
        JavaScript -: Coerce to numbers.
        
        "5" - 3 -> 2
        "hello" - 3 -> NaN
        """
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left - right
        # Coerce strings to numbers
        try:
            left_num = float(left) if isinstance(left, str) else left
            right_num = float(right) if isinstance(right, str) else right
            return left_num - right_num
        except (ValueError, TypeError):
            return float('nan')
    
    def binary_mul(self, left: Any, right: Any) -> Any:
        """
        JavaScript *: Coerce to numbers.
        
        "5" * 3 -> 15 (NOT "555" like you might expect!)
        "ab" * 3 -> NaN
        """
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left * right
        try:
            left_num = float(left) if isinstance(left, str) else left
            right_num = float(right) if isinstance(right, str) else right
            return left_num * right_num
        except (ValueError, TypeError):
            return float('nan')
    
    def binary_div(self, left: Any, right: Any) -> Any:
        """JavaScript /: Coerce to numbers."""
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left / right
        try:
            left_num = float(left) if isinstance(left, str) else left
            right_num = float(right) if isinstance(right, str) else right
            if right_num == 0:
                return float('inf') if left_num >= 0 else float('-inf')
            return left_num / right_num
        except (ValueError, TypeError):
            return float('nan')
    
    def binary_floor_div(self, left: Any, right: Any) -> Any:
        """JavaScript doesn't have //, use Math.floor(a/b)."""
        result = self.binary_div(left, right)
        if isinstance(result, float) and not (result != result):  # not NaN
            return int(result)
        return result
    
    def binary_mod(self, left: Any, right: Any) -> Any:
        """
        JavaScript %: Modulo with sign of DIVIDEND (opposite of Python!).
        
        -7 % 3 -> -1 (not 2)
        """
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            # Z3 modulo - need to adjust for JS semantics
            return left % right
        try:
            left_num = float(left) if isinstance(left, str) else left
            right_num = float(right) if isinstance(right, str) else right
            # Python's % has sign of divisor, JS has sign of dividend
            # JS: a % b = a - (b * trunc(a/b))
            import math
            return left_num - right_num * math.trunc(left_num / right_num)
        except (ValueError, TypeError, ZeroDivisionError):
            return float('nan')
    
    def binary_pow(self, left: Any, right: Any) -> Any:
        """JavaScript **: Exponentiation (ES2016+)."""
        try:
            left_num = float(left) if isinstance(left, str) else left
            right_num = float(right) if isinstance(right, str) else right
            return left_num ** right_num
        except (ValueError, TypeError):
            return float('nan')
    
    # =========================================================================
    # Comparison
    # =========================================================================
    
    def compare_eq(self, left: Any, right: Any) -> Any:
        """
        JavaScript ==: Loose equality with type coercion.
        
        1 == "1" -> true
        null == undefined -> true
        0 == false -> true
        """
        if isinstance(left, ExprRef) or isinstance(right, ExprRef):
            return left == right
        # Coercion rules are complex, simplified version:
        if type(left) == type(right):
            return left == right
        # String to number coercion
        try:
            if isinstance(left, str):
                return float(left) == right
            if isinstance(right, str):
                return left == float(right)
        except ValueError:
            pass
        return left == right
    
    def compare_strict_eq(self, left: Any, right: Any) -> Any:
        """
        JavaScript ===: Strict equality (no type coercion).
        
        1 === "1" -> false
        1 === 1 -> true
        """
        if isinstance(left, ExprRef) or isinstance(right, ExprRef):
            # In symbolic mode, types should match
            return left == right
        # No coercion: types must match
        if type(left) != type(right):
            return False
        return left == right
    
    def compare_lt(self, left: Any, right: Any) -> Any:
        """JavaScript <: With coercion."""
        return left < right
    
    def compare_le(self, left: Any, right: Any) -> Any:
        """JavaScript <=: With coercion."""
        return left <= right
    
    def compare_gt(self, left: Any, right: Any) -> Any:
        """JavaScript >: With coercion."""
        return left > right
    
    def compare_ge(self, left: Any, right: Any) -> Any:
        """JavaScript >=: With coercion."""
        return left >= right
    
    # =========================================================================
    # Boolean Operations
    # =========================================================================
    
    def bool_and(self, left: Any, right: Any) -> Any:
        """JavaScript &&: Short-circuit, returns operand value."""
        if isinstance(left, BoolRef) and isinstance(right, BoolRef):
            return And(left, right)
        if not self._is_truthy(left):
            return left
        return right
    
    def bool_or(self, left: Any, right: Any) -> Any:
        """JavaScript ||: Short-circuit, returns operand value."""
        if isinstance(left, BoolRef) and isinstance(right, BoolRef):
            return Or(left, right)
        if self._is_truthy(left):
            return left
        return right
    
    def bool_not(self, operand: Any) -> Any:
        """JavaScript !: Logical negation."""
        if isinstance(operand, BoolRef):
            return Not(operand)
        return not self._is_truthy(operand)
    
    def _is_truthy(self, value: Any) -> bool:
        """
        JavaScript truthiness rules.
        
        Falsy: null, undefined, false, 0, "", NaN
        Truthy: EVERYTHING ELSE including [] and {}!
        """
        if value is None:  # null/undefined
            return False
        if value is False:
            return False
        if value == 0:
            return False
        if value == "":
            return False
        if isinstance(value, float) and value != value:  # NaN check
            return False
        return True
    
    # =========================================================================
    # Type Coercion
    # =========================================================================
    
    def to_boolean(self, value: Any) -> Any:
        """Convert to boolean using JS truthiness rules."""
        if isinstance(value, BoolRef):
            return value
        return self._is_truthy(value)
    
    def to_string(self, value: Any) -> Any:
        """JavaScript String(): Convert to string."""
        if isinstance(value, SeqRef):
            return value
        if value is None:
            return "null"
        if value is True:
            return "true"
        if value is False:
            return "false"
        return str(value)
    
    def to_number(self, value: Any) -> Any:
        """
        JavaScript Number(): Convert to number.
        
        Returns NaN for invalid input (not exception).
        """
        if isinstance(value, ArithRef):
            return value
        if value is None:
            return 0  # Number(null) -> 0
        if value is True:
            return 1
        if value is False:
            return 0
        if isinstance(value, str):
            try:
                return float(value) if '.' in value else int(value)
            except ValueError:
                return float('nan')
        return float(value)
