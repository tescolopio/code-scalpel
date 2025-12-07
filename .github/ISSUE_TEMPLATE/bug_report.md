# Code Scalpel: GitHub Issues Template

This file contains templates for creating GitHub Issues from the product backlog.
Use these to populate the project board and track progress.

---

## How to Use

1. Copy each issue template below
2. Create a new GitHub issue
3. Paste the template
4. Add appropriate labels: `epic`, `feature`, `task`, `P0`, `P1`, `P2`, `P3`
5. Assign to project board
6. Link related issues

---

## EPIC Issues

### Epic 1: Core Package Infrastructure 🏗️

```markdown
**Epic Goal:** Establish a robust, production-ready Python package infrastructure

**Description:**
Transform the package structure from development prototype to production-ready with proper configuration, dependency management, and versioning.

**Features:**
- [ ] #X Fix Package Structure
- [ ] #X Dependency Management  
- [ ] #X Version Management

**Success Criteria:**
- Package installs with `pip install -e .`
- All modules importable as `from code_scalpel import X`
- Clean build to wheel and sdist
- Dependency management robust

**Priority:** P0
**Effort:** 3-4 days
**Dependencies:** None
```

### Epic 2: MCP Server Implementation 🔌

```markdown
**Epic Goal:** Implement complete MCP server exposing Code Scalpel capabilities

**Description:**
Build a Model Context Protocol (MCP) server that allows AI agents to access Code Scalpel's code analysis features through standardized tools, resources, and prompts.

**Features:**
- [ ] #X MCP Server Core
- [ ] #X MCP Tools - AST Analysis
- [ ] #X MCP Tools - PDG Analysis
- [ ] #X MCP Tools - Symbolic Execution
- [ ] #X MCP Resources
- [ ] #X MCP Prompts
- [ ] #X MCP Server CLI
- [ ] #X MCP Client Examples

**Success Criteria:**
- Server starts and responds to MCP messages
- All core tools functional and tested
- Examples work with Claude Desktop and Cursor
- Documentation comprehensive

**Priority:** P0
**Effort:** 10-12 days
**Dependencies:** Epic 1
```

### Epic 3: Testing Infrastructure 🧪

```markdown
**Epic Goal:** Achieve 80%+ test coverage with comprehensive testing

**Description:**
Build complete testing infrastructure with unit, integration, and E2E tests covering all functionality.

**Features:**
- [ ] #X Test Framework Setup
- [ ] #X Unit Tests - AST Tools
- [ ] #X Unit Tests - PDG Tools
- [ ] #X Unit Tests - Symbolic Execution
- [ ] #X Unit Tests - MCP Server
- [ ] #X Integration Tests
- [ ] #X Test Fixtures and Data

**Success Criteria:**
- 80%+ code coverage achieved
- All tests passing on Python 3.8-3.12
- Tests run in CI automatically
- Test fixtures comprehensive

**Priority:** P0
**Effort:** 12-14 days
**Dependencies:** Epic 1, Epic 2
```

---

## FEATURE Issues

### Feature 1.1: Fix Package Structure

```markdown
**Feature:** Fix Package Structure
**Epic:** Core Package Infrastructure

**Description:**
Restructure the package to follow Python packaging best practices and fix build issues.

**Problem:**
Current package structure doesn't match package name, causing build failures.

**Tasks:**
- [ ] Rename `src/` to `src/code_scalpel/`
- [ ] Update `pyproject.toml` with `[tool.hatch.build.targets.wheel]` section
- [ ] Create root `__init__.py` with version and exports
- [ ] Add `py.typed` marker file
- [ ] Update all internal imports
- [ ] Verify package builds: `python -m build`
- [ ] Test installation: `pip install -e .`

**Acceptance Criteria:**
- [x] Package installs successfully
- [x] All modules importable
- [x] Package builds to wheel and sdist
- [x] No import errors

**Priority:** P0
**Effort:** 2 days
**Dependencies:** None
**Labels:** `feature`, `P0`, `infrastructure`
```

### Feature 2.1: MCP Server Core

```markdown
**Feature:** MCP Server Core Implementation
**Epic:** MCP Server Implementation

**Description:**
Implement the core MCP server using FastMCP that handles protocol communication and provides the foundation for tools and resources.

**Tasks:**
- [ ] Add `fastmcp` and `mcp` dependencies
- [ ] Create `src/code_scalpel/mcp/` directory
- [ ] Implement `src/code_scalpel/mcp/server.py`
- [ ] Configure server metadata
- [ ] Implement stdio transport
- [ ] Add lifecycle management (startup/shutdown)
- [ ] Implement error handling
- [ ] Add structured logging
- [ ] Create configuration file support
- [ ] Write unit tests
- [ ] Document server configuration

**Acceptance Criteria:**
- [x] Server starts and stops cleanly
- [x] Responds to MCP protocol messages
- [x] Error handling comprehensive
- [x] Logging structured and useful
- [x] Configuration works via file/env vars
- [x] Tests passing

**Priority:** P0
**Effort:** 3 days
**Dependencies:** Feature 1.1
**Labels:** `feature`, `P0`, `mcp`
```

### Feature 2.2: MCP Tools - AST Analysis

```markdown
**Feature:** MCP Tools for AST Analysis
**Epic:** MCP Server Implementation

**Description:**
Expose AST analysis capabilities as MCP tools that AI agents can invoke.

**Tools to Implement:**
1. `parse_code_to_ast` - Parse code to AST
2. `analyze_code_structure` - Get functions, classes, metrics
3. `get_function_metrics` - Detailed function analysis
4. `get_class_metrics` - Detailed class analysis
5. `detect_code_smells` - Find code quality issues

**Tasks:**
- [ ] Implement `@mcp.tool` for each tool above
- [ ] Add Pydantic schemas for input validation
- [ ] Write comprehensive docstrings
- [ ] Add error handling
- [ ] Write unit tests for each tool
- [ ] Create integration test
- [ ] Document in MCP guide

**Acceptance Criteria:**
- [x] All tools callable via MCP
- [x] Tools return structured JSON
- [x] Input validation prevents errors
- [x] Documentation complete
- [x] Tests passing with 80%+ coverage

**Priority:** P0
**Effort:** 2 days
**Dependencies:** Feature 2.1
**Labels:** `feature`, `P0`, `mcp`, `ast`
```

---

## TASK Issues

### Task: Add fastmcp dependency

```markdown
**Task:** Add fastmcp dependency to project
**Feature:** MCP Server Core
**Epic:** MCP Server Implementation

**Description:**
Add `fastmcp` package to dependencies in `pyproject.toml`.

**Steps:**
1. Add `fastmcp>=0.1.0` to `dependencies` in `pyproject.toml`
2. Add `mcp[cli]>=1.0.0` to `dependencies`
3. Run `pip install -e .` to verify
4. Update `requirements.txt` if needed

**Acceptance Criteria:**
- [x] Dependencies added to `pyproject.toml`
- [x] Package installs without errors
- [x] Can import `from fastmcp import FastMCP`

**Priority:** P0
**Effort:** 0.25 days
**Labels:** `task`, `P0`, `dependencies`
```

### Task: Create MCP server directory structure

```markdown
**Task:** Create directory structure for MCP server
**Feature:** MCP Server Core
**Epic:** MCP Server Implementation

**Description:**
Set up the directory structure for MCP server implementation.

**Steps:**
1. Create `src/code_scalpel/mcp/` directory
2. Create `src/code_scalpel/mcp/__init__.py`
3. Create `src/code_scalpel/mcp/server.py` (empty)
4. Create `src/code_scalpel/mcp/tools/` directory
5. Create `src/code_scalpel/mcp/resources/` directory
6. Create `src/code_scalpel/mcp/prompts/` directory
7. Create `__init__.py` in each subdirectory

**Acceptance Criteria:**
- [x] Directory structure created
- [x] All `__init__.py` files present
- [x] Modules importable

**Priority:** P0
**Effort:** 0.25 days
**Labels:** `task`, `P0`, `infrastructure`
```

### Task: Write parse_code_to_ast MCP tool

```markdown
**Task:** Implement parse_code_to_ast MCP tool
**Feature:** MCP Tools - AST Analysis
**Epic:** MCP Server Implementation

**Description:**
Create MCP tool that parses source code to AST representation.

**Implementation:**
```python
@mcp.tool
def parse_code_to_ast(code: str, language: str = "python") -> dict:
    """Parse source code into an Abstract Syntax Tree.
    
    Args:
        code: Source code to parse
        language: Programming language (default: python)
        
    Returns:
        AST representation as JSON-serializable dict
    """
    # Implementation here
```

**Steps:**
1. Add tool function to `src/code_scalpel/mcp/tools/ast_tools.py`
2. Add Pydantic model for input validation
3. Implement AST parsing logic
4. Handle errors gracefully
5. Write unit tests
6. Document in docstring

**Acceptance Criteria:**
- [x] Tool function implemented
- [x] Input validation working
- [x] Returns valid JSON
- [x] Error handling comprehensive
- [x] Tests passing (coverage >80%)
- [x] Docstring complete

**Priority:** P0
**Effort:** 0.5 days
**Labels:** `task`, `P0`, `mcp`, `ast`
```

---

## Labels to Create

Create these labels in your GitHub repository:

### Type Labels
- `epic` - Large body of work (multiple features)
- `feature` - User-facing functionality
- `task` - Specific work item
- `bug` - Something isn't working
- `documentation` - Documentation improvements
- `testing` - Testing related

### Priority Labels
- `P0` - Critical (must have for production)
- `P1` - High (should have)
- `P2` - Medium (nice to have)
- `P3` - Low (future consideration)

### Category Labels
- `infrastructure` - Package, build, deployment
- `mcp` - MCP server related
- `ast` - AST analysis
- `pdg` - PDG analysis
- `symbolic-execution` - Symbolic execution
- `dependencies` - Dependency management
- `ci-cd` - CI/CD pipeline
- `security` - Security related
- `performance` - Performance optimization
- `multi-language` - Multi-language support

### Status Labels
- `blocked` - Blocked by dependencies
- `in-progress` - Currently being worked on
- `review-needed` - Ready for review
- `ready-to-merge` - Approved and ready

---

## GitHub Project Board Structure

### Column Layout
1. **Backlog** - All planned work
2. **Ready** - Ready to start (dependencies met)
3. **In Progress** - Currently being worked on
4. **Review** - In code review
5. **Done** - Completed and merged

### Views to Create
1. **By Epic** - Group by epic
2. **By Priority** - Group by P0, P1, P2, P3
3. **By Phase** - Group by roadmap phase
4. **Sprint View** - Current sprint items only

---

## Issue Numbering Convention

Use this format in issue titles:
- `[EPIC-1] Core Package Infrastructure`
- `[FEAT-1.1] Fix Package Structure`
- `[TASK-1.1.1] Rename src directory`

---

**Note:** This is a template file. Actual issues should be created in GitHub Issues and linked to the project board.

**Last Updated:** 2025-11-10
