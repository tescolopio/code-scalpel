# Code Scalpel Roadmap

> From Development to Production-Ready MCP-Enabled AI Code Analysis Toolkit

---

## Project Scope

> **⚠️ Scope Clarification:** Code Scalpel is a **developer toolkit and MCP server** for AI agents, NOT a mobile or desktop application. It is designed for programmatic use by AI coding assistants (Cursor, Cline, Claude Desktop), agentic frameworks (Autogen, CrewAI, Langchain), and CI/CD automation pipelines.

---

## Vision

Transform Code Scalpel into the industry-standard code analysis toolkit for AI agents, accessible via the Model Context Protocol (MCP) and supporting multiple programming languages.

---

## Current State (v0.1.0)

✅ **Implemented:**
- Core AST analysis tools for Python
- PDG (Program Dependence Graph) building and analysis
- Symbolic execution engine
- Basic AI agent integrations (Autogen, CrewAI, Langchain)
- Code parser infrastructure for multiple languages
- **NEW:** Stable `CodeAnalyzer` class with unified analysis pipeline
- **NEW:** MCP HTTP server (Flask-based) with `/analyze`, `/refactor`, `/security` endpoints
- **NEW:** CLI tool (`code-scalpel` command)
- **NEW:** Proper package structure (`pip install -e .` works)
- **NEW:** CI/CD pipeline (GitHub Actions)
- **NEW:** 151 passing tests (25% coverage)
- **NEW:** Getting Started & API Reference documentation

⏳ **In Progress:**
- PyPI publication
- Increase test coverage to 80%+

❌ **Missing:**
- Native MCP protocol (FastMCP) - current is HTTP/REST
- Multi-language support (beyond Python)
- Full documentation site

---

## Phase 1: Foundation (Weeks 1-2) 🏗️

**Goal:** Establish production-ready infrastructure and MCP server

### Week 1: Package Infrastructure
- [x] Fix package structure (rename `src/` → `src/code_scalpel/`) ✅ **Done 2025-12-03**
- [x] Configure proper `pyproject.toml` for build ✅ **Done 2025-12-03**
- [x] Manage dependencies (pin versions, create dev requirements) ✅ **Done 2025-12-03**
- [ ] Set up code formatting (Black, isort)
- [ ] Set up linting (Flake8, Pylint, mypy)
- [ ] Fix all linting and type errors

### Week 2: MCP Server Core
- [x] Implement MCP server (Flask HTTP) ✅ **Done 2025-12-03** (FastMCP deferred to v0.2.0)
- [x] Create AST analysis tools via HTTP endpoints ✅ **Done 2025-12-03**
- [x] Create PDG analysis tools via HTTP endpoints ✅ **Done 2025-12-03**
- [ ] Create symbolic execution tools (4 tools)
- [ ] Add MCP resources and prompts (for FastMCP)
- [x] Create CLI (`code-scalpel`) ✅ **Done 2025-12-03**
- [ ] Write MCP integration examples

**Deliverable:** Installable package with working MCP server ✅ **ACHIEVED**

---

## 📦 PyPI Alpha Release (v0.1.0a1) — Quick Path

**Goal:** Get a minimal, working package on PyPI for early testing

This is an accelerated path to release `pip install code-scalpel==0.1.0a1`:

### Prerequisites Checklist

- [x] **Package Structure** ✅ **Done 2025-12-03**
  - [x] Create `src/code_scalpel/` directory structure
  - [x] Add `__init__.py` with version and public API exports
  - [x] Ensure all modules have proper relative imports

- [ ] **pyproject.toml Updates**
  ```toml
  [project]
  name = "code-scalpel"
  version = "0.1.0a1"
  requires-python = ">=3.9"
  classifiers = [
      "Development Status :: 3 - Alpha",
      "Intended Audience :: Developers",
      "Topic :: Software Development :: Code Analyzers",
      "License :: OSI Approved :: MIT License",
      "Programming Language :: Python :: 3.9",
      "Programming Language :: Python :: 3.10",
      "Programming Language :: Python :: 3.11",
      "Programming Language :: Python :: 3.12",
      "Programming Language :: Python :: 3.13",
  ]
  
  [project.urls]
  Homepage = "https://github.com/tescolopio/code-scalpel"
  Documentation = "https://code-scalpel.readthedocs.io"
  Repository = "https://github.com/tescolopio/code-scalpel.git"
  Issues = "https://github.com/tescolopio/code-scalpel/issues"
  ```

- [x] **Dependencies** — Pin minimum versions in `pyproject.toml` ✅ **Done 2025-12-03**

- [x] **Build & Test Locally** ✅ **Done 2025-12-03**
  ```bash
  pip install build twine
  python -m build
  pip install dist/code_scalpel-0.1.0a1-py3-none-any.whl
  python -c "from code_scalpel import CodeAnalyzer; print('Success!')"
  ```

- [ ] **TestPyPI First**
  ```bash
  twine upload --repository testpypi dist/*
  pip install --index-url https://test.pypi.org/simple/ code-scalpel==0.1.0a1
  ```

- [ ] **Production PyPI**
  ```bash
  twine upload dist/*
  pip install code-scalpel==0.1.0a1
  ```

### Minimum Viable Alpha

For v0.1.0a1, the package should include:
- ✅ Core AST analysis tools
- ✅ Basic PDG functionality
- ✅ Working import structure
- ⏳ MCP server (can be v0.1.0a2)
- ⏳ Full test coverage (can be v0.1.0a2)

---

## Phase 2: Quality & Testing (Weeks 3-4) 🧪

**Goal:** Achieve 80%+ test coverage and comprehensive documentation

### Week 3: Testing Infrastructure
- [x] Set up pytest with coverage ✅ **Done 2025-12-03**
- [x] Write unit tests for AST tools ✅ **Done** (needs more coverage)
- [x] Write unit tests for PDG tools ✅ **Done** (needs more coverage)
- [ ] Write unit tests for symbolic execution (0% coverage)
- [x] Write unit tests for MCP server ✅ **Done**
- [x] Write integration tests ✅ **Done 2025-12-03**
- [ ] Create test fixtures for all languages

**Current Coverage:** 25% (Target: 80%)

### Week 4: Documentation
- [x] Write Getting Started guide ✅ **Done 2025-12-03**
- [x] Complete API reference documentation ✅ **Done 2025-12-03**
- [ ] Write MCP integration guide
- [ ] Update README with clear examples
- [ ] Write agent integration guides
- [ ] Create 10+ example scripts
- [ ] Set up Sphinx for auto-generated docs
- [ ] Write CONTRIBUTING.md

**Deliverable:** Well-tested, documented package ready for alpha release

---

## Phase 3: Production Readiness (Weeks 5-6) 🚀

**Goal:** Production-grade operations and public release

### Week 5: CI/CD & Operations
- [x] Set up GitHub Actions (testing, linting, docs) ✅ **Done 2025-12-03**
- [x] Configure multi-platform testing (Linux, macOS, Windows) ✅ **Done**
- [ ] Set up coverage reporting (Codecov) - configured but needs token
- [ ] Implement structured logging (JSON format)
- [ ] Add error handling and recovery
- [ ] Add monitoring/metrics (Prometheus format)
- [ ] Configure release automation
- [ ] Set up Dependabot

### Week 6: Security & Distribution
- [ ] Conduct security audit
- [ ] Fix all vulnerabilities
- [ ] Implement input validation
- [ ] Create SECURITY.md policy
- [ ] Publish to PyPI
- [ ] Test PyPI installation
- [ ] Create GitHub release (v0.2.0-alpha)
- [ ] Announce alpha release

**Deliverable:** Production-ready package on PyPI with automated CI/CD

---

## Phase 4: Enhancement (Weeks 7-9) ⚡

**Goal:** Multi-language support and performance optimization

### Week 7: Language Abstraction
- [ ] Design language-agnostic parser interface
- [ ] Create language registry system
- [ ] Implement language detection
- [ ] Document plugin system

### Week 8: JavaScript/TypeScript & Java
- [ ] Implement JS/TS parser and analyzers
- [ ] Implement Java parser and analyzers
- [ ] Create examples for both languages
- [ ] Test and document

### Week 9: Performance & More Languages
- [ ] Implement caching strategy
- [ ] Add parallel processing support
- [ ] Optimize memory usage
- [ ] Add C/C++, C#, Go support (basic)
- [ ] Performance benchmarking

**Deliverable:** Multi-language support with optimized performance (v0.3.0-beta)

---

## Phase 5: Community (Weeks 10+) 🌟

**Goal:** Build ecosystem and community

### Ongoing Activities
- [ ] Create project website
- [ ] Set up community templates (issues, PRs)
- [ ] Create showcase projects
- [ ] Publish to conda-forge
- [ ] Create Docker images
- [ ] Write blog posts and tutorials
- [ ] Present at conferences
- [ ] Engage community (Reddit, HN, Twitter)

**Deliverable:** Thriving community and ecosystem (v1.0.0)

---

## Version Milestones

### v0.2.0-alpha (Target: Week 6)
- ✨ MCP server implementation
- ✨ Core analysis tools working
- ✨ 80%+ test coverage
- ✨ Basic documentation
- ✨ Available on PyPI

### v0.3.0-beta (Target: Week 9)
- ✨ JavaScript/TypeScript support
- ✨ Java support
- ✨ Performance optimizations
- ✨ Comprehensive documentation
- ✨ Stable MCP API

### v1.0.0 (Target: Week 12)
- ✨ Production-ready for all features
- ✨ 5+ languages supported
- ✨ Battle-tested in production
- ✨ Strong community presence
- ✨ Complete documentation
- ✨ High performance and scalability

---

## Key Success Metrics

### Technical Excellence
- **Test Coverage:** 80%+ maintained
- **Performance:** <2s for 1000 LOC Python file
- **MCP Response:** <5s (95th percentile)
- **Security:** Zero critical vulnerabilities
- **Compatibility:** Python 3.8-3.12 supported

### Adoption & Growth
- **Downloads:** 1000+/month on PyPI
- **Stars:** 100+ on GitHub
- **Contributors:** 10+ community contributors
- **Production:** 5+ known production deployments
- **Documentation:** 95%+ pages complete

### Quality & Reliability
- **CI Success:** 95%+ pass rate
- **Issue Response:** <1 day average
- **PR Review:** <7 days average
- **Uptime:** 99.9%+ (for hosted demos)

---

## Focus Areas

### Top 3 Priorities
1. **MCP Integration:** Make Code Scalpel the best code analysis tool for AI agents
2. **Documentation:** Ensure anyone can use Code Scalpel effectively
3. **Quality:** Production-grade code with comprehensive testing

### Key Differentiators
- 🔌 **MCP-First Design:** Native integration with AI agent ecosystem
- 🔍 **Deep Analysis:** AST, PDG, and Symbolic Execution in one tool
- 🌍 **Multi-Language:** Support for major programming languages
- 🚀 **Production-Ready:** Built for scale and reliability
- 🤖 **AI-Optimized:** Designed specifically for AI agent workflows

---

## Dependencies & Prerequisites

### Required Before v1.0
1. Stable MCP protocol spec from Anthropic
2. Community feedback on API design
3. Real-world usage in production environments
4. Performance validation on large codebases

### External Dependencies
- **MCP Protocol:** Track Anthropic releases
- **Python Support:** Follow Python release schedule
- **Security:** Monitor dependency vulnerabilities
- **Community:** Engage early adopters and contributors

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| MCP spec changes | High | Version MCP interface, stay close to releases |
| Dependency vulnerabilities | High | Automated scanning, rapid response |
| Poor performance at scale | Medium | Early benchmarking, optimization priority |
| Low adoption | Medium | Focus on docs, examples, outreach |
| Multi-language complexity | Medium | Abstract interfaces, incremental approach |

---

## Communication Plan

### Updates
- **Weekly:** Progress updates in GitHub Discussions
- **Bi-weekly:** Blog posts on key milestones
- **Monthly:** Community calls (after v0.2.0)

### Channels
- **GitHub:** Issues, PRs, Discussions
- **Documentation:** docs.code-scalpel.dev (future)
- **Social:** Twitter, Reddit, Dev.to
- **Events:** Conference talks, meetups

---

## How to Contribute

See the [PRODUCT_BACKLOG.md](PRODUCT_BACKLOG.md) for detailed tasks and the [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon) for contribution guidelines.

**Priority areas for contributors:**
1. MCP server tools and resources
2. Multi-language parser implementations
3. Documentation and examples
4. Performance optimizations
5. Test coverage improvements

---

**Last Updated:** 2025-12-03
**Version:** 1.2 (User Testing Ready)
**Next Review:** After PyPI Alpha Release

---

## Recent Accomplishments (2025-12-03)

### Branches Merged
- `copilot/stabilize-codeanalyzer-class` - Stable CodeAnalyzer API
- `copilot/add-unit-integration-tests` - CI pipeline + 151 tests
- `copilot/implement-autogen-scalpel-wrapper` - MCP server + AI integrations

### Key Deliverables
- ✅ `pip install -e .` works
- ✅ `code-scalpel` CLI functional
- ✅ MCP HTTP server at `/analyze`, `/refactor`, `/security`
- ✅ 151 tests passing
- ✅ Documentation: getting_started.md, api_reference.md
