# Code Scalpel Roadmap

> From Development to Production-Ready MCP-Enabled AI Code Analysis Toolkit

---

## Project Scope

> **Scope Clarification:** Code Scalpel is a **developer toolkit and MCP server** for AI agents, NOT a mobile or desktop application. It is designed for programmatic use by AI coding assistants (Cursor, Cline, Claude Desktop), agentic frameworks (Autogen, CrewAI, Langchain), and CI/CD automation pipelines.

---

## Vision

Transform Code Scalpel into the industry-standard code analysis toolkit for AI agents, accessible via the Model Context Protocol (MCP) and supporting multiple programming languages.

---

## Current State (v0.3.0) RELEASED

**Released:** 2024-12-06 ("The Mathematician" Release)  
**PyPI:** https://pypi.org/project/code-scalpel/0.3.0/  
**GitHub Tag:** https://github.com/tescolopio/code-scalpel/releases/tag/v0.3.0

**Production Features:**

- Core AST analysis tools for Python (94% coverage)
- PDG (Program Dependence Graph) building and analysis (86% coverage)
- MCP HTTP server (Flask-based) with `/analyze`, `/refactor`, `/security` endpoints
- CLI tool (`code-scalpel` command)
- AI agent integrations (Autogen, CrewAI, Langchain)
- Proper package structure (`pip install code-scalpel` works)
- CI/CD pipeline (GitHub Actions)
- 469 passing tests (76% coverage on active code)
- Security audit passed (Gate 0)
- Clean artifact build (Gate 1)

**v0.3.0 "The Mathematician" Features:**

- **String Support** - Z3 String theory integration
  - String literals and concatenation in symbolic execution
  - `StringSort()` in type inference and state management
  - String constraints in ConstraintSolver
- **Security Analysis (Beta)** - Taint-based vulnerability detection
  - `SecurityAnalyzer` for end-to-end security scanning
  - `TaintTracker` for data flow tracking
  - SQL Injection detection (CWE-89)
  - XSS detection (CWE-79)
  - Command Injection detection (CWE-78)
  - Path Traversal detection (CWE-22)
  - Convenience functions: `find_sql_injections()`, `find_xss()`, etc.
- **469 tests**, 76% coverage

**Known Limitations:**

- Symbolic Execution: Int/Bool/String types only
- No floating point support yet
- Loops bounded to 10 iterations
- Security Analysis not yet validated against OWASP Benchmark

**Not Yet Implemented:**

- Native MCP protocol (FastMCP) - current is HTTP/REST
- Multi-language support (beyond Python)
- Full documentation site

---

## Previous Releases

### v0.2.0 "Redemption" (2024-12-06)

- Symbolic Execution working (Int/Bool only)
- 426 tests, 76% coverage
- Z3 solver integration

### v0.1.0 (2024-12-04)

- Initial PyPI release
- MCP HTTP server
- AST and PDG tools

---

## Phase 1: Foundation (Weeks 1-2) COMPLETE

**Goal:** Establish production-ready infrastructure and MCP server

- [x] Package structure and build configuration
- [x] MCP HTTP server with endpoints
- [x] CLI tool (`code-scalpel`)
- [x] Security audit and artifact verification
- [x] **PyPI Release v0.1.0** 2024-12-04

---

## Phase 2: Quality and Testing (Weeks 3-4) COMPLETE

**Goal:** Achieve 80%+ test coverage and comprehensive documentation

- [x] pytest with coverage (76% achieved)
- [x] Unit tests for AST, PDG, Symbolic Execution
- [x] Integration tests
- [x] Getting Started guide and API reference
- [x] **PyPI Release v0.2.0 and v0.3.0** 2024-12-06

---

## Phase 3: Security Hardening (Weeks 5-6) - v0.3.1

**Goal:** Validate and harden security analysis tools

### Week 5: Security Validation

- [ ] Validate TaintTracker against OWASP Benchmark (SQLi subset)
- [ ] Validate TaintTracker against OWASP Benchmark (XSS subset)
- [ ] Add false-positive rate metrics
- [ ] Add false-negative rate metrics
- [ ] Document known detection gaps

### Week 6: Security and Distribution

- [ ] Conduct security audit of Code Scalpel itself
- [ ] Fix all vulnerabilities found
- [ ] Implement input validation on MCP endpoints
- [ ] Create SECURITY.md policy
- [ ] Set up Dependabot for vulnerability alerts

**Deliverable:** Security-validated analysis tools (v0.3.1)

---

## Phase 4: The Polyglot (Weeks 7-9) - v0.4.0

**Goal:** JavaScript/TypeScript support (focused scope)

### Week 7: Language Abstraction

- [ ] Design language-agnostic parser interface
- [ ] Create language registry system
- [ ] Implement language detection from file extension

### Week 8: JavaScript/TypeScript Support

- [ ] Integrate tree-sitter-javascript
- [ ] Implement JS/TS AST analyzer
- [ ] Implement JS/TS PDG builder (basic)
- [ ] Test: `code-scalpel analyze app.js` works

### Week 9: Testing and Polish

- [ ] Write tests for JS/TS support
- [ ] Performance benchmarking
- [ ] Documentation for multi-language usage

**Deliverable:** JavaScript/TypeScript support (v0.4.0 "The Polyglot")

---

## Phase 5: The Speedster (Weeks 10-12) - v0.5.0

**Goal:** Native MCP protocol and performance optimization

### Week 10: FastMCP Integration

- [ ] Evaluate FastMCP vs current Flask implementation
- [ ] Design migration path (backward compatible)
- [ ] Implement FastMCP transport layer
- [ ] Maintain HTTP fallback for compatibility

### Week 11-12: Performance Optimization

- [ ] Implement caching strategy (AST/PDG caching)
- [ ] Add parallel processing for multi-file analysis
- [ ] Optimize memory usage for large codebases
- [ ] Performance benchmarking (target: <2s for 1000 LOC)

**Deliverable:** Production-grade MCP server (v0.5.0 "The Speedster")

---

## Phase 6: Expansion (Weeks 13+) - v0.6.0+

**Goal:** Additional languages and community growth

### Java Support (Alpha)

- [ ] Integrate tree-sitter-java or javalang
- [ ] Basic AST analysis for Java
- [ ] Note: Full PDG for Java is complex - scope appropriately

### Community Building

- [ ] Create project website
- [ ] Publish to conda-forge
- [ ] Create Docker images

**Deliverable:** Multi-language toolkit (v1.0.0)

---

## Version Milestones

| Version | Codename | Status | Key Features |
|---------|----------|--------|--------------|
| v0.1.0 | - | RELEASED | MCP HTTP server, AST/PDG, CLI |
| v0.2.0 | Redemption | RELEASED | Symbolic Execution (Int/Bool) |
| v0.3.0 | The Mathematician | RELEASED | String support, Security Analysis |
| v1.0.0 | The Standard | RELEASED | API freeze, caching, Z3 hardening, 654 tests |
| v1.0.1 | - | RELEASED | Docker port alignment, documentation fixes |
| v1.0.2 | - | RELEASED | Documentation & Resource Hardening, 760 tests |
| v1.1.0 | The Polyglot | PLANNED | Extended JavaScript/TypeScript support |
| v1.2.0 | The Speedster | PLANNED | FastMCP, performance optimization |

---

## Key Success Metrics

### Technical Excellence

- **Test Coverage:** 80%+ maintained
- **Performance:** <2s for 1000 LOC Python file
- **MCP Response:** <5s (95th percentile)
- **Security:** Zero critical vulnerabilities

### Security Analysis Quality

- **OWASP SQLi Detection:** >80% true positive rate
- **False Positive Rate:** <20%
- **Supported CWEs:** 4+ (89, 79, 78, 22)

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| MCP spec changes | High | Version MCP interface, maintain HTTP fallback |
| Multi-language complexity | Medium | Incremental approach, tree-sitter abstraction |
| Security false positives | Medium | OWASP benchmark validation |
| Java AST complexity | Medium | Scope as "Alpha", use existing libraries |

---

## How to Contribute

See [PRODUCT_BACKLOG.md](PRODUCT_BACKLOG.md) for detailed tasks.

**Priority areas:**

1. OWASP Benchmark test cases
2. JavaScript/TypeScript parser integration
3. Documentation and examples
4. Performance optimizations

---

**Last Updated:** 2025-01-21  
**Version:** 5.0 (v1.0.2 "Documentation & Resource Hardening" Released)  
**Next Milestone:** v1.1.0 The Polyglot

---

## Release History

| Version | Date | Highlights |
|---------|------|------------|
| v1.0.2 | 2025-01-21 | Documentation & Resource Hardening - MCP resources documented, 760 tests |
| v1.0.1 | 2025-01-20 | Docker port alignment, documentation fixes |
| v1.0.0 | 2025-01-20 | "The Standard" - API freeze, caching, Z3 hardening, 654 tests |
| v0.3.0 | 2024-12-06 | "The Mathematician" - String support, Security Analysis, 469 tests |
| v0.2.0 | 2024-12-06 | "Redemption" - Symbolic Execution works, 426 tests |
| v0.1.0 | 2024-12-04 | First public release on PyPI |
