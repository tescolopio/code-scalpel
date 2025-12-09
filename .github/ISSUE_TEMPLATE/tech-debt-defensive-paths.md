---
name: Tech Debt - Defensive Path Coverage
about: Track untested defensive code paths in symbolic execution
title: "[TECH DEBT] Verify Defensive Paths in ir_interpreter and security modules"
labels: tech-debt, testing, priority-medium
assignees: ''
---

## Summary

v1.1.0 shipped with 99% coverage on symbolic execution core files, but **13 partial branches remain untested**. These are not "theoretical" - they are real code paths that handle edge cases.

## The Problem

The team labeled these as "defensive None checks" and "loop edge cases that require specific rare inputs." This is rationalization, not engineering.

**If your code handles `None`, you must test what happens when `None` occurs.**

## Specific Gaps

### ir_interpreter.py (7 partial branches)

| Line | Branch | What It Guards |
|------|--------|----------------|
| 775->770 | `elif not state.has_variable(name)` | Assignment when `value_expr` is None but variable exists |
| 807->826 | `if right is not None and self._semantics is not None` | AugAssign with None operand |
| 860->866 | `if self._semantics is not None` | If statement without semantics |
| 862->866 | `if bool_cond is not None` | Boolean conversion returns None |
| 942->947 | While loop semantic check | While condition evaluation fails |
| 944->947 | While bool conversion | Boolean conversion returns None |
| 1227->1234 | Loop/iteration edge case | TBD - needs investigation |

### security_analyzer.py (4 partial branches)

| Line | Branch | What It Guards |
|------|--------|----------------|
| 322->317 | Loop over args | Early return before loop exhausts |
| 325->317 | `elif isinstance(arg, ast.BinOp)` | BinOp arg not encountered |
| 327->325 | Nested loop over vars | Early return before inner loop exhausts |
| 416->420 | `while isinstance(current, ast.Attribute)` | Loop exhaustion case |

### taint_tracker.py (2 partial branches)

| Line | Branch | What It Guards |
|------|--------|----------------|
| 378->382 | `if sink_set` | Empty sink set after parsing |
| 398->408 | `for _ in range(10)` | Config file search exhausts 10 levels |

## Required Actions

### 1. Write Explicit Tests for None Cases
```python
def test_assignment_with_none_value_existing_variable():
    """When value_expr evaluates to None but variable already exists."""
    # Setup: create state with existing variable
    # Execute: assign expression that returns None
    # Assert: variable retains old value (or whatever the correct behavior is)
```

### 2. Test Loop Exhaustion
```python
def test_find_config_file_deep_nesting():
    """Config search when nested > 10 directories with no pyproject.toml."""
    # Setup: mock os.getcwd to return deeply nested path
    # Execute: _find_config_file()
    # Assert: returns None without crashing
```

### 3. Fix or Remove Pragmas
The `# pragma: no branch` comments are not working. Either:
- The syntax is wrong (should be on the `if` line, not inside the block)
- The code IS reachable and the pragma is a lie

Audit each pragma and either fix placement or remove it and write a test.

## Acceptance Criteria

- [ ] All 13 branches either tested OR proven unreachable with correct pragma
- [ ] No "theoretical" justifications - concrete test cases or proof of unreachability
- [ ] Coverage report shows 100% or explicit exclusions documented

## Priority

Medium - Not blocking, but creates risk of production edge case failures.

## References

- PR that introduced this debt: (link to v1.1.0 cleanup PR)
- Coverage report: `pytest --cov=code_scalpel.symbolic_execution_tools --cov-branch`
