# RFC-001: Symbolic Execution Engine

**Status:** Draft  
**Authors:** Code Scalpel Team  
**Created:** 2024-12-04  
**Target Version:** v0.2.0 "Redemption"

---

## 1. Executive Summary

This RFC proposes the architecture for a production-ready symbolic execution engine for Code Scalpel. The current implementation is non-functional (quarantined in v0.1.0) due to missing core components: type inference, state management, and Z3 integration.

**Goal:** Enable Code Scalpel to answer: *"What inputs cause this code path to execute?"*

---

## 2. Background

### 2.1 What is Symbolic Execution?

Symbolic execution runs a program with **symbolic values** instead of concrete values. Instead of `x = 5`, we say `x = α` (some unknown). As execution proceeds, we collect **path conditions**—constraints that must be true for execution to reach a given point.

```python
def example(x):
    if x > 10:      # Path condition: α > 10
        if x < 20:  # Path condition: α > 10 ∧ α < 20
            bug()   # Reachable if ∃α: (α > 10 ∧ α < 20)
```

An SMT solver (Z3) determines if the path conditions are satisfiable and provides concrete values (e.g., `x = 15`) that trigger the path.

### 2.2 Why the Current Implementation Failed

The v0.1.0 symbolic execution module has:

| Component | Status | Problem |
|-----------|--------|---------|
| `SymbolicExecutionEngine` | Exists | `_infer_type()` returns `NotImplementedError` |
| `ConstraintSolver` | Exists | `solve()` returns `NotImplementedError` |
| `PathExplorer` | Exists | Depends on above, so unusable |
| Z3 Integration | Missing | No actual SMT queries |

**Root Cause:** The code was scaffolded without understanding the compiler design required to translate Python → SMT.

---

## 3. Proposed Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python Source Code                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AST Parser (existing)                        │
│                  code_scalpel.ast_tools                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 TYPE INFERENCE ENGINE (NEW)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Type        │  │ Type        │  │ Type        │             │
│  │ Collector   │  │ Unifier     │  │ Annotator   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  Input: AST                                                     │
│  Output: {var_name: Z3Type} mapping                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 SYMBOLIC STATE MANAGER (NEW)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Symbolic    │  │ Path        │  │ Memory      │             │
│  │ Variables   │  │ Conditions  │  │ Model       │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  Tracks: Current values, constraints, heap/stack state          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 AST INTERPRETER (NEW)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Expression  │  │ Statement   │  │ Control     │             │
│  │ Evaluator   │  │ Executor    │  │ Flow        │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  Walks AST, updates SymbolicState, forks on branches            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Z3 SOLVER BRIDGE (NEW)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Constraint  │  │ Z3          │  │ Model       │             │
│  │ Translator  │  │ Interface   │  │ Extractor   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  Converts SymbolicState → Z3 formulas, queries satisfiability   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ANALYSIS RESULTS                           │
│  - Reachable paths with concrete trigger values                 │
│  - Unreachable (dead) code detection                            │
│  - Potential crash conditions                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Details

#### 3.2.1 Type Inference Engine

**Purpose:** Determine Z3 types for Python variables.

**Strategy:** Flow-sensitive type inference with fallback to `BitVec(64)` for unknowns.

```python
class TypeInferenceEngine:
    """
    Infers Z3 types from Python AST.
    
    Supported mappings:
    - int literals, int() calls → z3.Int
    - bool literals, comparisons → z3.Bool  
    - str literals, str() calls → z3.String
    - float → z3.Real (limited support)
    - bytes → z3.BitVec
    - Unknown → z3.BitVec(64) with warning
    """
    
    def infer(self, ast_node: ast.AST) -> Dict[str, z3.SortRef]:
        """Returns {variable_name: z3_sort} mapping."""
        pass
```

**Type Inference Rules:**

| Python Pattern | Z3 Type | Confidence |
|----------------|---------|------------|
| `x = 42` | `Int` | High |
| `x = True` | `Bool` | High |
| `x = "hello"` | `String` | High |
| `x = 3.14` | `Real` | Medium |
| `x = int(input())` | `Int` | Medium |
| `x = some_function()` | `BitVec(64)` | Low (warning) |
| `x = obj.attr` | Unsupported | Error |

#### 3.2.2 Symbolic State Manager

**Purpose:** Track the "universe" at any execution point.

```python
@dataclass
class SymbolicState:
    """Immutable snapshot of symbolic execution state."""
    
    # Variable bindings: name → Z3 expression
    variables: Dict[str, z3.ExprRef]
    
    # Path condition: conjunction of constraints to reach this state
    path_condition: List[z3.BoolRef]
    
    # Program counter: current AST node
    pc: ast.AST
    
    # Call stack for function tracking
    call_stack: List[CallFrame]
    
    # Heap model (for objects/lists - Phase 2)
    heap: Optional[HeapModel] = None
    
    def fork(self, condition: z3.BoolRef) -> Tuple['SymbolicState', 'SymbolicState']:
        """Fork state for branching: returns (true_branch, false_branch)."""
        true_state = self.with_constraint(condition)
        false_state = self.with_constraint(z3.Not(condition))
        return true_state, false_state
```

#### 3.2.3 AST Interpreter

**Purpose:** Walk the AST symbolically, updating state.

```python
class SymbolicInterpreter:
    """Interprets Python AST with symbolic values."""
    
    def execute(self, node: ast.AST, state: SymbolicState) -> List[SymbolicState]:
        """
        Execute node, return list of resulting states.
        (Multiple states possible due to branching)
        """
        match node:
            case ast.Assign():
                return self._handle_assign(node, state)
            case ast.If():
                return self._handle_if(node, state)  # Forks state
            case ast.While():
                return self._handle_while(node, state)  # Loop handling
            case ast.Return():
                return self._handle_return(node, state)
            # ... etc
```

#### 3.2.4 Z3 Solver Bridge

**Purpose:** Interface between our symbolic state and Z3.

```python
class Z3Bridge:
    """Translates symbolic state to Z3 and queries satisfiability."""
    
    def __init__(self, timeout_ms: int = 5000):
        self.solver = z3.Solver()
        self.solver.set("timeout", timeout_ms)
    
    def is_satisfiable(self, state: SymbolicState) -> Tuple[bool, Optional[Dict]]:
        """
        Check if state's path condition is satisfiable.
        Returns (is_sat, model_if_sat).
        """
        self.solver.push()
        for constraint in state.path_condition:
            self.solver.add(constraint)
        
        result = self.solver.check()
        
        if result == z3.sat:
            model = self._extract_model(self.solver.model(), state.variables)
            self.solver.pop()
            return True, model
        
        self.solver.pop()
        return False, None
    
    def _extract_model(self, z3_model, variables) -> Dict[str, Any]:
        """Convert Z3 model to Python values."""
        return {name: self._z3_to_python(z3_model[expr]) 
                for name, expr in variables.items()}
```

---

## 4. Scope & Limitations

### 4.1 Phase 1 Scope (v0.2.0)

**Supported:**
- Integer arithmetic (`+`, `-`, `*`, `//`, `%`)
- Boolean logic (`and`, `or`, `not`)
- Comparisons (`<`, `>`, `<=`, `>=`, `==`, `!=`)
- Simple `if/elif/else` branches
- Simple `while` loops (bounded unrolling, default 10 iterations)
- Function calls to analyzed functions (inlined)
- Integer and boolean variables

**Not Supported (Phase 1):**
- Strings (complex in SMT)
- Lists, dicts, objects (requires heap model)
- External function calls (mocked as unconstrained)
- Floating point (Z3 Real is slow)
- Exceptions
- Recursion (limited depth)
- Generators, async

### 4.2 Known Hard Problems

| Problem | Strategy |
|---------|----------|
| **Path Explosion** | Bounded exploration (max 1000 states), priority queue by depth |
| **Loop Unrolling** | Default 10 iterations, configurable, warn on potential infinite |
| **External Calls** | Return unconstrained symbolic value, log warning |
| **Timeouts** | Per-query timeout (5s), total timeout (60s) |
| **Memory** | State limit, garbage collect unreachable states |

---

## 5. API Design

### 5.1 Public API

```python
from code_scalpel.symbolic_execution_tools import SymbolicExecutor

# Create executor
executor = SymbolicExecutor(
    max_depth=100,           # Max path depth
    loop_bound=10,           # Max loop iterations
    timeout_seconds=60,      # Total analysis timeout
    solver_timeout_ms=5000,  # Per-query Z3 timeout
)

# Analyze code
result = executor.analyze("""
def check_password(password: int) -> bool:
    if password == 12345:
        return True  # SECRET PATH
    return False
""")

# Get results
for path in result.paths:
    print(f"Path to line {path.end_line}:")
    print(f"  Reachable: {path.is_reachable}")
    print(f"  Trigger: {path.trigger_values}")  # {'password': 12345}
    print(f"  Constraints: {path.path_condition}")
```

### 5.2 Integration with CodeAnalyzer

```python
from code_scalpel import CodeAnalyzer

analyzer = CodeAnalyzer()
results = analyzer.analyze(code, include_symbolic=True)

# Symbolic results integrated with other analysis
print(results.symbolic.reachable_paths)
print(results.symbolic.dead_code)  # Paths with unsat conditions
print(results.symbolic.crash_conditions)
```

---

## 6. Implementation Plan

### 6.1 Milestones

| Milestone | Description | Deliverable |
|-----------|-------------|-------------|
| **M0** | Tracer Bullet | `scripts/z3_hello_world.py` works |
| **M1** | Type Inference | `TypeInferenceEngine` for int/bool |
| **M2** | State Manager | `SymbolicState` with fork/clone |
| **M3** | Basic Interpreter | Handle Assign, BinOp, Compare, If |
| **M4** | Z3 Bridge | `is_satisfiable()` works end-to-end |
| **M5** | Loop Support | Bounded `while` loop unrolling |
| **M6** | Integration | Wire into `CodeAnalyzer`, remove quarantine |
| **M7** | Testing | 80% coverage on symbolic_execution_tools |

### 6.2 Estimated Effort

| Phase | Effort | Risk |
|-------|--------|------|
| M0-M1 | 2 hours | Low |
| M2-M3 | 4 hours | Medium |
| M4 | 2 hours | Low |
| M5 | 3 hours | Medium (loop edge cases) |
| M6-M7 | 4 hours | Low |
| **Total** | ~15 hours | |

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
def test_type_inference_int_literal():
    engine = TypeInferenceEngine()
    types = engine.infer(ast.parse("x = 42"))
    assert types["x"] == z3.IntSort()

def test_symbolic_if_branch():
    executor = SymbolicExecutor()
    result = executor.analyze("if x > 10: y = 1")
    assert len(result.paths) == 2  # True branch, False branch
    
def test_unreachable_code():
    executor = SymbolicExecutor()
    result = executor.analyze("""
        if False:
            unreachable()
    """)
    assert result.paths[0].is_reachable == False
```

### 7.2 Integration Tests

```python
def test_password_cracker():
    """The classic symbolic execution demo."""
    code = '''
    def check(password):
        if password == 42:
            return "ACCESS GRANTED"
        return "DENIED"
    '''
    executor = SymbolicExecutor()
    result = executor.analyze(code)
    
    granted_path = [p for p in result.paths if "GRANTED" in str(p.return_value)]
    assert granted_path[0].trigger_values == {"password": 42}
```

---

## 8. Open Questions

1. **String Support:** Should we use Z3's String theory (slow) or defer to Phase 2?
2. **Object Heap:** How do we model `obj.attr` assignments?
3. **External Calls:** Mock with fresh symbolic, or require annotations?
4. **Error Reporting:** How verbose should "unsupported construct" warnings be?

---

## 9. References

- [Z3 Python API](https://z3prover.github.io/api/html/namespacez3py.html)
- [KLEE: Symbolic Execution Engine](https://klee.github.io/)
- [A Survey of Symbolic Execution Techniques](https://arxiv.org/abs/1610.00502)
- [Python Type Inference Paper](https://dl.acm.org/doi/10.1145/2983990.2984017)

---

## 10. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2024-12-04 | Phase 1: Int/Bool only | Reduce scope, prove architecture |
| 2024-12-04 | Bounded loop unrolling (10) | Avoid infinite loops without invariants |
| 2024-12-04 | External calls → unconstrained | Safer than crashing, user can annotate later |

---

**Next Step:** Implement M0 (Tracer Bullet) to prove Z3 integration works.
