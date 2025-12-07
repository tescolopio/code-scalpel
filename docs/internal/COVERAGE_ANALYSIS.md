# Coverage Deep Dive Analysis

Generated: 2025-01-21 | Code Scalpel v1.0.2

## Executive Summary

**Current Coverage: 71%** (target: 75%)  
**Tests: 760 passing**  
**Statements: 6137 total, 1513 missed**  

---

## Coverage by Module

### Fully Covered (90%+)

| Module | Coverage | Notes |
|--------|----------|-------|
| `ast_tools/analyzer.py` | 94% | Core analysis stable |
| `ast_tools/call_graph.py` | 98% | NEW in v1.0.2 |
| `ast_tools/dependency_parser.py` | 97% | NEW in v1.0.2 |
| `taint_tracker.py` | 89% | Security analysis |
| `generators/refactor_simulator.py` | 87% | Refactoring tools |

### Well Covered (75-89%)

| Module | Coverage | Notes |
|--------|----------|-------|
| `pdg_tools/builder.py` | 86% | PDG construction |
| `pdg_tools/analyzer.py` | 82% | PDG analysis |
| `constraint_solver.py` | 82% | Z3 constraint solving |
| `generators/test_generator.py` | 79% | Test generation |
| `ir/normalizers/python_normalizer.py` | 79% | Python IR |
| `code_analyzer.py` | 77% | High-level analysis API |
| `symbolic_execution/engine.py` | 76% | Symbolic execution core |
| `symbolic_execution/state_manager.py` | 76% | State management |
| `symbolic_execution/type_inference.py` | 76% | Type inference |
| `utilities/cache.py` | 78% | Caching layer |

### Moderate Coverage (50-74%)

| Module | Coverage | Notes |
|--------|----------|-------|
| `ir_interpreter.py` | 70% | IR interpretation |
| `security_analyzer.py` | 70% | Security analysis |
| `autogen.py` | 70% | AutoGen integration |
| `base_parser.py` | 68% | Parser base |
| `mcp/server.py` | 67% | MCP server |
| `rest_api_server.py` | 66% | REST API |
| `ir/normalizers/javascript_normalizer.py` | 60% | JS/TS support |
| `ir/normalizers/java_normalizer.py` | 53% | Java support |
| `crewai.py` | 51% | CrewAI integration |

### Needs Improvement (<50%)

| Module | Coverage | Priority |
|--------|----------|----------|
| `cli.py` | 42% | MEDIUM - CLI commands |
| `pdg_tools/slicer.py` | 39% | LOW - Advanced feature |
| `ir/semantics.py` | 45% | LOW - IR semantics |
| `ir/tree_sitter_visitor.py` | 47% | LOW - Tree-sitter |

---

## Progress Since v0.1.0

| Metric | v0.1.0 | v1.0.0 | v1.0.2 | Change |
|--------|--------|--------|--------|--------|
| Coverage | 25% | 68% | 71% | +46% |
| Tests | ~100 | 654 | 760 | +660 |
| Statements Tested | 1689 | 4178 | 4624 | +2935 |

---

## v1.0.2 Test Additions

| Test File | Tests Added | Coverage Impact |
|-----------|-------------|-----------------|
| `test_call_graph.py` | 36 tests | call_graph.py 0%→98% |
| `test_dependency_parser.py` | 41 tests | dependency_parser.py 0%→97% |
| `test_mcp_resources.py` | 29 tests | MCP resources verified |

---

## Recommended Next Steps

### Priority 1: CLI Coverage (42%)
- Add tests for `scan`, `analyze`, `mcp` commands
- Test error handling and output formatting

### Priority 2: Integration Tests
- Real (non-mock) integration tests for AutoGen, CrewAI
- End-to-end MCP server tests

### Priority 3: IR Coverage
- `ir/semantics.py` - expand semantic operation tests
- `ir/tree_sitter_visitor.py` - tree-sitter edge cases

---

**Last Updated:** 2025-01-21
**Version:** v1.0.2
