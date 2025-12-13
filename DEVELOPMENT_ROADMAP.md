# Code Scalpel Development Roadmap

**Document Version:** 1.3  
**Last Updated:** December 12, 2025  
**Current Release:** v1.4.0 (Stable)  
**Maintainer:** 3D Tech Solutions LLC

---

## Executive Summary

Code Scalpel is an **MCP server toolkit designed for AI agents** (Claude, GitHub Copilot, Cursor, etc.) to perform surgical code operations without hallucination risk. By providing AI assistants with precise, AST-based code analysis and modification tools, we eliminate the guesswork that leads to broken code, incorrect line numbers, and context loss.

### Core Mission

**Enable AI agents to work on real codebases with surgical precision.**

Traditional AI coding assistants struggle with:
- **Hallucinated line numbers** - AI guesses where code is located
- **Context overflow** - Large files exceed token limits, AI loses track
- **Blind modifications** - AI rewrites entire functions when only one line needs changing
- **No verification** - AI cannot confirm its changes preserve behavior

Code Scalpel solves these by giving AI agents MCP tools that:
- **Extract exactly what's needed** - Surgical extraction of functions/classes by name, not line guessing
- **Modify without collateral damage** - Replace specific symbols, preserving surrounding code
- **Verify before applying** - Simulate refactors to detect behavior changes
- **Analyze with certainty** - Real AST parsing, not regex pattern matching

### Current State (v1.4.0)

| Metric | Value | Status |
|--------|-------|--------|
| MCP Tools | 10 tools (analyze, extract, security, test gen, context) | Stable |
| Test Suite | 1,692 tests passing | Stable |
| Code Coverage | 95%+ | Target Met |
| Security Detection | 17+ vulnerability types, 30+ secret patterns | Stable |
| Languages | Python (full), JS/Java (structural) | Expanding |
| AI Agent Integrations | Claude Desktop, VS Code Copilot | Verified |

### Target State

| Metric | Target | Milestone |
|--------|--------|-----------|
| MCP Tools | 15+ tools | v2.1.0 |
| Languages | Python, TypeScript, JavaScript, Java | v2.0.0 |
| Cross-File Operations | Full project context | v1.5.1 |
| AI Verification | Behavior-preserving refactor check | v2.1.0 |
| Auto-Fix Generation | AI-verified security fixes | v2.1.0 |

---

## Release Timeline

```
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ v1.3.0  │  │ v1.4.0  │  │ v1.5.0  │  │ v1.5.1  │  │ v2.0.0  │  │ v2.1.0  │
│ Harden  │─>│ Context │─>│ Project │─>│ Cross-  │─>│ Poly-   │─>│ AI      │
│         │  │         │  │ Intel   │  │ File    │  │ glot    │  │ Verify  │
└─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘
     │            │            │            │            │            │
   Path Res    More Vuln   Dep Graph    Import Res   TypeScript   Behavior
   Secrets     Patterns    Call Graph   Taint Flow   JavaScript   Verify
   Coverage    SSTI/XXE    Project Map  Multi-File   Java         Auto-Fix
```

## v1.3.0 - "Hardening"

### Overview

**Theme:** Stability and Security Coverage  
**Goal:** Fix critical blockers, expand detection to 95%+  
**Effort:** ~10 developer-days  
**Risk Level:** Low (incremental improvements)

### Priorities

| Priority | Feature                                 | Owner | Effort | Dependencies |
| -------- | --------------------------------------- | ----- | ------ | ------------ |
| **P0**   | Fix `extract_code` file path resolution |TDE   | 2 days | None         |
| **P0**   | Add hardcoded secret detection          | TDE   | 1 day  | None         |
| **P0**   | Add NoSQL injection (MongoDB)           | TDE   | 1 day  | None         |
| **P0**   | Add LDAP injection sinks                | TDE   | 1 day  | None         |
| **P0**   | Surgical tools → 95% coverage           | TDE   | 3 days | None         |
| **P1**   | Line numbers in all MCP tools           | TDE   | 1 day  | None         |
| **P1**   | Improve test generation types           | TDE   | 2 days | None         |

### Technical Specifications

#### 1. Fix `extract_code` File Path Resolution

**Problem:** External testers reported `"File not found: test_code_scalpel_security.py"` when using relative paths.

**Root Cause:** The `extract_code` tool doesn't resolve paths relative to the workspace root.

**Solution:**

```python
# In src/code_scalpel/mcp/server.py or surgical tools

def resolve_file_path(file_path: str, workspace_root: str = None) -> str:
    """Resolve file path to absolute path."""
    path = Path(file_path)

    # Already absolute
    if path.is_absolute():
        return str(path)

    # Try relative to workspace root
    if workspace_root:
        workspace_path = Path(workspace_root) / path
        if workspace_path.exists():
            return str(workspace_path)

    # Try relative to current working directory
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return str(cwd_path)

    # Try common project structures
    for prefix in ["src", "lib", "app", "."]:
        candidate = Path(prefix) / path
        if candidate.exists():
            return str(candidate.resolve())

    raise FileNotFoundError(f"Cannot resolve path: {file_path}")
```

**Acceptance Criteria:**

- [x] `extract_code("utils.py", ...)` works from project root
- [x] `extract_code("src/utils.py", ...)` works with relative paths
- [x] `extract_code("/absolute/path/utils.py", ...)` works unchanged
- [x] Clear error message when file truly doesn't exist

#### 2. Hardcoded Secret Detection

**New Vulnerability Type:** `HARDCODED_SECRET` (CWE-798)

**Patterns to Detect:**

```python
# src/code_scalpel/symbolic_execution_tools/taint_tracker.py

HARDCODED_SECRET_PATTERNS = {
    "aws_access_key": r"(?i)AKIA[A-Z0-9]{16}",
    "aws_secret_key": r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*['\"][A-Za-z0-9/+=]{40}['\"]",
    "github_token": r"ghp_[a-zA-Z0-9]{36}",
    "github_oauth": r"gho_[a-zA-Z0-9]{36}",
    "github_app": r"ghu_[a-zA-Z0-9]{36}",
    "gitlab_token": r"glpat-[a-zA-Z0-9\-]{20,}",
    "stripe_live": r"sk_live_[a-zA-Z0-9]{24,}",
    "stripe_test": r"sk_test_[a-zA-Z0-9]{24,}",
    "slack_token": r"xox[baprs]-[a-zA-Z0-9\-]{10,}",
    "slack_webhook": r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[a-zA-Z0-9]+",
    "google_api": r"AIza[0-9A-Za-z\-_]{35}",
    "firebase": r"AAAA[A-Za-z0-9_-]{7}:[A-Za-z0-9_-]{140}",
    "twilio_sid": r"AC[a-z0-9]{32}",
    "twilio_token": r"SK[a-z0-9]{32}",
    "sendgrid": r"SG\.[a-zA-Z0-9\-_]{22}\.[a-zA-Z0-9\-_]{43}",
    "private_key": r"-----BEGIN\s+(RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----",
    "generic_secret": r"(?i)(secret|password|passwd|pwd|token|api[_-]?key)\s*[=:]\s*['\"][^'\"]{8,}['\"]",
}
```

**Implementation:**

```python
# Add to SecuritySink enum
class SecuritySink(Enum):
    # ... existing sinks ...
    HARDCODED_SECRET = "hardcoded_secret"

# Add detection in security_analyzer.py
def _check_hardcoded_secrets(self, node: ast.AST) -> List[Vulnerability]:
    """Check for hardcoded secrets in string literals."""
    vulnerabilities = []

    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            for secret_type, pattern in HARDCODED_SECRET_PATTERNS.items():
                if re.search(pattern, child.value):
                    vulnerabilities.append(Vulnerability(
                        type="Hardcoded Secret",
                        cwe="CWE-798",
                        severity="HIGH",
                        message=f"Hardcoded {secret_type} detected",
                        line=child.lineno,
                        column=child.col_offset,
                    ))

    return vulnerabilities
```

**Test Cases:**

```python
def test_detects_aws_access_key():
    code = 'AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"'
    result = security_scan(code)
    assert len(result.vulnerabilities) == 1
    assert "aws_access_key" in result.vulnerabilities[0].message.lower()

def test_detects_github_token():
    code = 'GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"'
    result = security_scan(code)
    assert len(result.vulnerabilities) == 1

def test_ignores_placeholder():
    code = 'API_KEY = "your-api-key-here"'  # Placeholder, not real
    result = security_scan(code)
    assert len(result.vulnerabilities) == 0  # Or flag as "potential"
```

#### 3. NoSQL Injection (MongoDB)

**New Sink Category:** MongoDB query methods

**Patterns:**

```python
# Add to SINK_PATTERNS in taint_tracker.py

"nosql_injection": [
    # PyMongo
    "collection.find",
    "collection.find_one",
    "collection.find_one_and_delete",
    "collection.find_one_and_replace",
    "collection.find_one_and_update",
    "collection.aggregate",
    "collection.count_documents",
    "collection.distinct",
    "collection.update_one",
    "collection.update_many",
    "collection.delete_one",
    "collection.delete_many",
    "collection.insert_one",
    "collection.insert_many",
    "collection.replace_one",
    "db.command",
    # Motor (async)
    "motor_collection.find",
    "motor_collection.find_one",
    "motor_collection.aggregate",
    # MongoEngine
    "Document.objects",
    "QuerySet.filter",
    "QuerySet.get",
],
```

**Vulnerable Pattern Example:**

```python
# VULNERABLE - user input directly in query
@app.route('/user/<user_id>')
def get_user(user_id):
    # NoSQL injection: {"$gt": ""} returns all users
    user = db.users.find_one({"_id": user_id})  # SINK
    return jsonify(user)

# SAFE - validated ObjectId
from bson import ObjectId
@app.route('/user/<user_id>')
def get_user_safe(user_id):
    try:
        oid = ObjectId(user_id)  # Validates format
        user = db.users.find_one({"_id": oid})
        return jsonify(user)
    except:
        return "Invalid ID", 400
```

#### 4. LDAP Injection

**New Sink Category:** LDAP query methods

**Patterns:**

```python
# Add to SINK_PATTERNS in taint_tracker.py

"ldap_injection": [
    # python-ldap
    "ldap.search",
    "ldap.search_s",
    "ldap.search_st",
    "ldap.search_ext",
    "ldap.search_ext_s",
    "ldap.bind",
    "ldap.bind_s",
    "ldap.simple_bind",
    "ldap.simple_bind_s",
    "ldap.modify",
    "ldap.modify_s",
    "ldap.add",
    "ldap.add_s",
    "ldap.delete",
    "ldap.delete_s",
    # ldap3
    "Connection.search",
    "Connection.bind",
    "Connection.modify",
    "Connection.add",
    "Connection.delete",
],
```

**Vulnerable Pattern Example:**

```python
# VULNERABLE - user input in LDAP filter
def authenticate(username, password):
    ldap_filter = f"(&(uid={username})(userPassword={password}))"  # INJECTION!
    conn.search("dc=example,dc=com", ldap_filter)

# SAFE - escaped input
from ldap3.utils.conv import escape_filter_chars
def authenticate_safe(username, password):
    safe_user = escape_filter_chars(username)
    safe_pass = escape_filter_chars(password)
    ldap_filter = f"(&(uid={safe_user})(userPassword={safe_pass}))"
    conn.search("dc=example,dc=com", ldap_filter)
```

### Acceptance Criteria Checklist

v1.3.0 Release Criteria:

[x] extract_code works from project root (P0) - path_resolution.py
[x] extract_code works with relative paths (P0) - path_utils.py
[x] extract_code works with absolute paths (P0) - path_utils.py
[x] extract_code provides clear error for missing files (P0) - FileNotFoundError

[x] Detects AWS access keys (P0) - 30+ patterns in taint*tracker.py
[x] Detects AWS secret keys (P0) - HARDCODED_SECRET_PATTERNS
[x] Detects GitHub tokens (ghp*, gho*, ghu*) (P0) - All 3 formats
[x] Detects Stripe keys (sk*live*, sk*test*) (P0) - Both formats
[x] Detects private keys (-----BEGIN PRIVATE KEY-----) (P0) - RSA/EC/DSA/OPENSSH
[x] Detects generic secrets (password=, api_key=) (P0) - With placeholder filter

[x] Detects MongoDB find() with tainted input (P0) - nosql_injection sinks
[x] Detects MongoDB aggregate() with tainted input (P0) - PyMongo + Motor
[x] Detects MongoDB update/delete with tainted input (P0) - Full CRUD coverage

[x] Detects LDAP search with tainted filter (P0) - ldap_injection sinks
[x] Detects LDAP bind with tainted credentials (P0) - python-ldap + ldap3

[x] SurgicalExtractor coverage >= 95% (P0) - 95%
[x] SurgicalPatcher coverage >= 95% (P0) - 96%

[x] All MCP tools return line numbers (P1) - FunctionInfo/ClassInfo models
[x] Test generation infers float types correctly (P1) - FLOAT type + RealSort

[x] All 1,669+ tests passing (Gate) - 1,669 passed
[x] No regressions in existing detections (Gate) - Verified
[x] Documentation updated (Gate) - README, copilot-instructions, RELEASE_NOTES_v1.3.0

## v1.4.0 - "Context" ✅ RELEASED

### Overview

**Theme:** Enhanced AI Context and Detection Coverage  
**Goal:** Give AI agents richer context about code and expand vulnerability detection  
**Effort:** ~12 developer-days  
**Risk Level:** Low (extends existing MCP tools)  
**Status:** ✅ Released December 12, 2024

### Priorities

| Priority | Feature | Owner | Effort | Status |
|----------|---------|-------|--------|--------|
| **P0** | `get_file_context` MCP tool | TDE | 3 days | ✅ Done |
| **P0** | `get_symbol_references` MCP tool | TDE | 2 days | ✅ Done |
| **P0** | XXE detection (CWE-611) | TDE | 2 days | ✅ Done |
| **P0** | SSTI detection (CWE-1336) | TDE | 1 day | ✅ Done |
| **P1** | JWT vulnerabilities | - | 2 days | Deferred to v1.5.0 |
| **P1** | Mass assignment detection | - | 2 days | Deferred to v1.5.0 |

### Technical Specifications

#### 1. `get_file_context` MCP Tool

**Purpose:** AI agents need to understand a file's role without reading the entire file.

```python
# New MCP tool for AI agents
async def get_file_context(file_path: str) -> FileContext:
    """Provide AI with file overview without full content."""
    return FileContext(
        file_path=file_path,
        language="python",
        line_count=450,
        functions=["main", "process_request", "validate_input"],
        classes=["RequestHandler", "Validator"],
        imports=["flask", "sqlalchemy", "os"],
        exports=["RequestHandler", "main"],
        complexity_score=12,
        has_security_issues=True,
        summary="Flask request handler with database operations"
    )
```

**Why AI Agents Need This:**
- Quickly assess if a file is relevant to their task
- Understand file structure without consuming tokens on full content
- Make informed decisions about which functions to extract

#### 2. `get_symbol_references` MCP Tool

**Purpose:** AI agents need to find all usages of a function/class before modifying it.

```python
# New MCP tool for AI agents  
async def get_symbol_references(
    symbol_name: str, 
    project_root: str
) -> SymbolReferences:
    """Find all references to a symbol across the project."""
    return SymbolReferences(
        symbol_name="validate_input",
        definition_file="src/validators.py",
        definition_line=42,
        references=[
            Reference(file="src/handlers.py", line=15, context="validate_input(request.data)"),
            Reference(file="src/api.py", line=88, context="if validate_input(payload):"),
            Reference(file="tests/test_validators.py", line=12, context="assert validate_input(...)"),
        ],
        total_references=3
    )
```

**Why AI Agents Need This:**
- Safe refactoring - know all call sites before changing signature
- Impact analysis - understand blast radius of changes
- No hallucination - real references, not guessed ones

#### 3. XXE Detection (XML External Entity)

**CWE:** CWE-611

**Vulnerable Parsers:**

```python
"xxe": [
    # Vulnerable by default
    "xml.etree.ElementTree.parse",
    "xml.etree.ElementTree.fromstring",
    "xml.etree.ElementTree.iterparse",
    "xml.dom.minidom.parse",
    "xml.dom.minidom.parseString",
    "xml.sax.parse",
    "xml.sax.parseString",
    "lxml.etree.parse",
    "lxml.etree.fromstring",
    "lxml.etree.XML",
    "xmlrpc.client.ServerProxy",
],

# Safe alternatives (sanitizers)
"xxe_safe": [
    "defusedxml.parse",
    "defusedxml.fromstring",
    "defusedxml.ElementTree.parse",
    "defusedxml.minidom.parse",
],
```

#### 2. SSTI Detection (Server-Side Template Injection)

**CWE:** CWE-1336

**Vulnerable Patterns:**

```python
"ssti": [
    # Jinja2
    "jinja2.Template",
    "Environment.from_string",
    "Template.render",  # When template comes from user
    # Mako
    "mako.template.Template",
    # Django (when template string is user-controlled)
    "django.template.Template",
    # Tornado
    "tornado.template.Template",
],
```

**Example:**

```python
# VULNERABLE
@app.route('/render')
def render_template():
    template = request.args.get('template')
    return jinja2.Template(template).render()  # RCE!

# SAFE - use file-based templates
@app.route('/render')
def render_safe():
    return render_template('page.html', data=request.args.get('data'))
```

### Acceptance Criteria Checklist

v1.4.0 Release Criteria:

[x] get_file_context: Returns file overview without full content (P0)
[x] get_file_context: Lists functions, classes, imports (P0)
[x] get_file_context: Reports complexity score (P0)
[x] get_file_context: Flags files with security issues (P0)

[x] get_symbol_references: Finds all usages across project (P0)
[x] get_symbol_references: Returns file, line, and context snippet (P0)
[x] get_symbol_references: Works for functions, classes, variables (P0)
[x] get_symbol_references: Performance < 5s for 100-file project (P0)

[x] XXE: Detects xml.etree.ElementTree.parse with tainted input (P0)
[x] XXE: Detects xml.dom.minidom.parse with tainted input (P0)
[x] XXE: Detects lxml.etree.parse with tainted input (P0)
[x] XXE: Recognizes defusedxml.* as safe sanitizers (P0)

[x] SSTI: Detects jinja2.Template with user-controlled string (P0)
[x] SSTI: Detects Environment.from_string injection (P0)
[x] SSTI: Detects mako.template.Template injection (P0)

[x] Agents: Base agent framework with MCP tool integration (P0)
[x] Agents: Code review agent implementation (P0)
[x] Agents: Security agent implementation (P0)
[x] Agents: Optimization agent implementation (P0)

DEFERRED TO v1.5.0 - JWT: Detects algorithm confusion vulnerabilities (P1)
DEFERRED TO v1.5.0 - JWT: Detects missing signature verification (P1)
DEFERRED TO v1.5.0 - Mass Assignment: Detects unfiltered request.json usage (P1)

[x] MCP tools registered and documented (Gate)
[x] All tests passing (Gate)
[x] Code coverage >= 95% (Gate)
[x] No regressions in v1.3.0 detections (Gate)

---

## v1.5.0 - "Project Intelligence"

### Overview

**Theme:** Project-Wide Understanding for AI Agents  
**Goal:** Give AI agents complete project context without reading every file  
**Effort:** ~10 developer-days  
**Risk Level:** Low (uses existing PDG infrastructure)

### Priorities

| Priority | Feature | Owner | Effort | Dependencies |
|----------|---------|-------|--------|--------------|
| **P0** | `get_project_map` MCP tool | TBD | 3 days | None |
| **P0** | `get_call_graph` MCP tool | TBD | 2 days | PDG exists |
| **P0** | `scan_dependencies` MCP tool | TBD | 3 days | None |
| **P1** | Circular dependency detection | TBD | 1 day | PDG exists |
| **P1** | JWT vulnerabilities | TBD | 2 days | None |
| **P1** | Mass assignment detection | TBD | 2 days | None |

### Technical Specifications

#### 1. `get_project_map` MCP Tool

**Purpose:** AI agents need a mental model of the entire project structure.

```python
# New MCP tool for AI agents
async def get_project_map(project_root: str) -> ProjectMap:
    """Provide AI with complete project structure."""
    return ProjectMap(
        project_root=project_root,
        total_files=47,
        total_lines=12500,
        languages={"python": 42, "yaml": 3, "json": 2},
        entry_points=["src/main.py", "src/cli.py"],
        modules=[
            Module(path="src/handlers/", purpose="HTTP request handlers", files=8),
            Module(path="src/models/", purpose="Database models", files=6),
            Module(path="src/utils/", purpose="Utility functions", files=4),
        ],
        key_files=[
            KeyFile(path="src/config.py", purpose="Configuration management"),
            KeyFile(path="src/database.py", purpose="Database connection"),
        ],
        dependency_count=23,
        test_coverage=87.5
    )
```

**Why AI Agents Need This:**
- Understand project architecture without exploring randomly
- Identify where to make changes based on purpose, not guessing
- Know which modules are related before making cross-cutting changes

#### 2. `get_call_graph` MCP Tool

**Purpose:** AI agents need to understand function relationships.

```python
# New MCP tool for AI agents
async def get_call_graph(
    entry_point: str,
    depth: int = 3
) -> CallGraph:
    """Generate call graph from entry point."""
    return CallGraph(
        entry_point="main",
        nodes=[
            Node(name="main", file="src/main.py", line=10),
            Node(name="process_request", file="src/handlers.py", line=25),
            Node(name="validate_input", file="src/validators.py", line=42),
        ],
        edges=[
            Edge(caller="main", callee="process_request"),
            Edge(caller="process_request", callee="validate_input"),
        ],
        mermaid_diagram="graph TD\\n  main --> process_request\\n  ...",
    )
```

**Why AI Agents Need This:**
- Trace execution flow to understand code behavior
- Find all functions affected by a change
- Identify dead code or unused functions

#### 3. `scan_dependencies` MCP Tool

**Purpose:** AI agents need to know about vulnerable dependencies.

```python
# New MCP tool for AI agents
async def scan_dependencies(requirements_path: str) -> DependencyReport:
    """Scan dependencies for known CVEs."""
    return DependencyReport(
        total_dependencies=23,
        vulnerable_count=2,
        vulnerabilities=[
            CVE(package="requests", version="2.25.0", cve="CVE-2023-32681", 
                severity="HIGH", fixed_in="2.31.0"),
        ]
    )
```

### Acceptance Criteria Checklist

v1.5.0 Release Criteria:

[ ] get_project_map: Returns complete project structure (P0)
[ ] get_project_map: Identifies entry points automatically (P0)
[ ] get_project_map: Groups files into logical modules (P0)
[ ] get_project_map: Reports language breakdown (P0)
[ ] get_project_map: Performance < 10s for 500-file project (P0)

[ ] get_call_graph: Traces calls from entry point (P0)
[ ] get_call_graph: Returns nodes with file/line info (P0)
[ ] get_call_graph: Generates Mermaid diagram (P0)
[ ] get_call_graph: Handles recursive calls (P0)
[ ] get_call_graph: Respects depth limit (P0)

[ ] scan_dependencies: Parses requirements.txt (P0)
[ ] scan_dependencies: Parses pyproject.toml (P0)
[ ] scan_dependencies: Queries OSV API for CVEs (P0)
[ ] scan_dependencies: Returns severity levels (P0)
[ ] scan_dependencies: Suggests fixed versions (P0)

[ ] Circular Deps: Detects direct circular imports (P1)
[ ] Circular Deps: Reports cycle path clearly (P1)

[ ] New MCP tools registered and documented (Gate)
[ ] All tests passing (Gate)
[ ] Code coverage >= 95% (Gate)
[ ] No regressions in v1.4.0 detections (Gate)

---

## v1.5.1 - "CrossFile"

### Overview

**Theme:** Multi-File Operations for AI Agents  
**Goal:** Enable AI agents to understand and modify code across file boundaries  
**Effort:** ~15 developer-days  
**Risk Level:** High (architectural complexity)

### Priorities

| Priority | Feature | Owner | Effort | Dependencies |
|----------|---------|-------|--------|--------------|
| **P0** | `extract_cross_file` MCP tool | TBD | 5 days | Import resolution |
| **P0** | Cross-file taint tracking | TBD | 5 days | Import resolution |
| **P0** | Import resolution engine | TBD | 5 days | None |

### Why AI Agents Need Cross-File Operations

**Problem:** AI agents today work file-by-file. When a function in `utils.py` is called from `handlers.py`, the AI has no way to:
1. Know what callers exist before changing a signature
2. Track if user input flows across file boundaries
3. Safely refactor code that spans multiple files

**Solution:** New MCP tools that operate at project scope.

### Technical Specifications

#### 1. `extract_cross_file` MCP Tool

```python
# New MCP tool for AI agents
async def extract_cross_file(
    symbol_name: str,
    project_root: str,
    include_callers: bool = True,
    include_callees: bool = True
) -> CrossFileExtraction:
    """Extract a symbol with all its cross-file dependencies."""
    return CrossFileExtraction(
        target=SymbolCode(name="get_user", file="models.py", code="def get_user(...)"),
        callers=[
            SymbolCode(name="handle_request", file="views.py", code="def handle_request(...)"),
        ],
        callees=[
            SymbolCode(name="execute_query", file="database.py", code="def execute_query(...)"),
        ],
        import_chain=["views.py imports models", "models.py imports database"],
    )
```

#### 2. Cross-File Taint Tracking

```
Challenge: Track taint across files

File: views.py                    File: models.py
─────────────                     ─────────────
def handle_request(req):          def get_user(user_id):
    user_id = req.args['id']  ──────>  query = f"SELECT * FROM users WHERE id={user_id}"
    return get_user(user_id)           cursor.execute(query)  # VULNERABLE!
```

**Solution: Inter-Procedural Analysis**

```python
class CrossFileTaintTracker:
    def __init__(self, project_root: str):
        self.import_graph = {}  # module -> imports
        self.function_signatures = {}  # func -> (params, return_taint)

    def analyze_project(self, entry_point: str):
        # Phase 1: Build import graph
        self.build_import_graph(entry_point)

        # Phase 2: Analyze each module
        for module in topological_sort(self.import_graph):
            self.analyze_module(module)

        # Phase 3: Propagate taint across calls
        self.propagate_cross_file_taint()
```

**Scope Limitations (v1.5.1):**
- Single-hop imports only (direct `from x import y`)
- No dynamic imports (`importlib.import_module`)
- No `sys.path` manipulation
- No circular import resolution (fail gracefully)

### Acceptance Criteria Checklist

v1.5.1 Release Criteria:

[ ] extract_cross_file: Extracts symbol with callers (P0)
[ ] extract_cross_file: Extracts symbol with callees (P0)
[ ] extract_cross_file: Returns import chain (P0)
[ ] extract_cross_file: Works across 3+ files (P0)

[ ] Import Resolution: Resolves "from module import func" (P0)
[ ] Import Resolution: Resolves "import module" (P0)
[ ] Import Resolution: Resolves relative imports (P0)
[ ] Import Resolution: Handles __init__.py packages (P0)
[ ] Import Resolution: Returns clear error for missing modules (P0)

[ ] Cross-File Taint: Tracks taint through function calls (P0)
[ ] Cross-File Taint: Tracks taint through return values (P0)
[ ] Cross-File Taint: Detects SQL injection across 2 files (P0)
[ ] Cross-File Taint: Detects command injection across 2 files (P0)
[ ] Cross-File Taint: Reports source file and sink file (P0)
[ ] Cross-File Taint: Reports full taint propagation path (P0)

[ ] Builds import graph for project (P0)
[ ] Topological sort handles acyclic dependencies (P0)
[ ] Graceful failure on circular imports (P0)
[ ] Performance: Analyzes 50-file project in < 30s (P0)

[ ] All tests passing (Gate)
[ ] Code coverage >= 95% (Gate)
[ ] No regressions in v1.5.0 detections (Gate)
[ ] Cross-file taint documented with examples (Gate)

---

## v2.0.0 - "Polyglot"

### Overview

**Theme:** Multi-Language MCP Tools for AI Agents  
**Goal:** Enable AI agents to work surgically on TypeScript, JavaScript, and Java projects  
**Effort:** ~25 developer-days  
**Risk Level:** High (new language architecture)

### Why Polyglot Matters for AI Agents

AI agents today are asked to work on full-stack projects: Python backends, TypeScript frontends, Java microservices. Without language-aware surgical tools, agents must:
- Guess at code structure based on text patterns
- Risk breaking syntax when modifying unfamiliar languages
- Miss language-specific vulnerabilities

**Solution:** Extend all MCP tools to support TypeScript, JavaScript, and Java with the same surgical precision as Python.

### Priorities

| Priority | Feature | Owner | Effort | Dependencies |
|----------|---------|-------|--------|--------------|
| **P0** | TypeScript/JavaScript AST support | TBD | 10 days | tree-sitter |
| **P0** | `extract_code` for TS/JS/Java | TBD | 5 days | AST support |
| **P0** | `security_scan` for TS/JS/Java | TBD | 8 days | AST support |
| **P1** | Java Spring security patterns | TBD | 5 days | tree-sitter |
| **P1** | JSX/TSX support | TBD | 3 days | TS support |

### Technical Specifications

#### 1. Multi-Language `extract_code`

```python
# Extended MCP tool
async def extract_code(
    file_path: str = None,
    code: str = None,
    target_type: str,  # "function", "class", "method", "interface", "type"
    target_name: str,
    language: str = "auto"  # "python", "typescript", "javascript", "java", "auto"
) -> ContextualExtractionResult:
    """Surgically extract code in any supported language."""
    # Auto-detect language from file extension or content
    # Use tree-sitter for TS/JS/Java parsing
    # Return same structured result regardless of language
```

**Why This Matters:**
- AI agents can use ONE tool for all languages
- Consistent interface reduces agent confusion
- No hallucinated line numbers regardless of language

#### 2. JavaScript/TypeScript Vulnerabilities

```python
JS_SINK_PATTERNS = {
    # DOM XSS
    "dom_xss": [
        "innerHTML",
        "outerHTML",
        "document.write",
        "document.writeln",
        "insertAdjacentHTML",
    ],

    # Eval Injection
    "eval_injection": [
        "eval",
        "Function",
        "setTimeout",  # with string arg
        "setInterval",  # with string arg
        "new Function",
    ],

    # Prototype Pollution
    "prototype_pollution": [
        "Object.assign",
        "_.merge",
        "_.extend",
        "$.extend",
        "lodash.merge",
    ],

    # Node.js Injection
    "node_injection": [
        "child_process.exec",
        "child_process.execSync",
        "child_process.spawn",
        "require",  # with user input
    ],

    # SQL Injection (Node.js)
    "node_sql": [
        "connection.query",
        "pool.query",
        "knex.raw",
        "sequelize.query",
    ],
}
```

### Acceptance Criteria Checklist

v2.0.0 Release Criteria:

[ ] extract_code: Works for TypeScript functions/classes (P0)
[ ] extract_code: Works for JavaScript functions/classes (P0)
[ ] extract_code: Works for Java methods/classes (P0)
[ ] extract_code: Auto-detects language from file extension (P0)

[ ] TypeScript AST: Parses .ts files correctly (P0)
[ ] TypeScript AST: Parses .tsx files correctly (P0)
[ ] TypeScript AST: Handles type annotations (P0)
[ ] TypeScript AST: Handles interfaces and types (P0)

[ ] JavaScript AST: Parses .js files correctly (P0)
[ ] JavaScript AST: Parses .jsx files correctly (P0)
[ ] JavaScript AST: Handles ES6+ syntax (P0)
[ ] JavaScript AST: Handles CommonJS and ESM imports (P0)

[ ] security_scan: Detects DOM XSS (innerHTML, document.write) (P0)
[ ] security_scan: Detects eval injection (P0)
[ ] security_scan: Detects prototype pollution (P0)
[ ] security_scan: Detects Node.js command injection (P0)
[ ] security_scan: Detects Node.js SQL injection (P0)

[ ] Java: Parses .java files correctly (P1)
[ ] Java: Detects SQL injection in JPA queries (P1)
[ ] Java: Detects command injection (P1)

[ ] All MCP tools work identically across languages (Gate)
[ ] All tests passing (Gate)
[ ] Code coverage >= 95% (Gate)
[ ] No regressions in Python detections (Gate)

---

## v2.1.0 - "AI Verify"

### Overview

**Theme:** Behavior Verification for AI-Generated Code  
**Goal:** Enable AI agents to verify their changes don't break existing behavior  
**Effort:** ~25 developer-days  
**Risk Level:** High (safety-critical)

### Why AI Agents Need Verification

The biggest risk of AI-assisted coding is **silent breakage**: the AI makes a change that looks correct but subtly breaks existing behavior. Currently, AI agents have no way to verify their changes are safe.

**Solution:** MCP tools that let AI agents verify behavior preservation before applying changes.

### Priorities

| Priority | Feature | Owner | Effort | Dependencies |
|----------|---------|-------|--------|--------------|
| **P0** | `verify_behavior` MCP tool | TBD | 10 days | simulate_refactor |
| **P0** | `suggest_fix` MCP tool | TBD | 8 days | security_scan |
| **P0** | `apply_verified_fix` MCP tool | TBD | 5 days | verify_behavior |
| **P1** | Batch verification for multi-file changes | TBD | 5 days | cross-file |

### Technical Specifications

#### 1. `verify_behavior` MCP Tool

**Purpose:** AI agents need to verify their changes don't break existing behavior.

```python
# New MCP tool for AI agents
async def verify_behavior(
    original_code: str,
    modified_code: str,
    test_inputs: list[dict] = None
) -> BehaviorVerification:
    """Verify that modified code preserves original behavior."""
    return BehaviorVerification(
        is_safe=True,
        confidence=0.95,
        behavior_preserved=True,
        changes_detected=[
            Change(type="signature_same", description="Function signature unchanged"),
            Change(type="return_type_same", description="Return type unchanged"),
        ],
        warnings=[],
        recommendation="Safe to apply"
    )
```

**Why AI Agents Need This:**
- Confidence before applying changes
- Catch subtle bugs that text-based diffs miss
- Prevent "it compiles but doesn't work" failures

#### 2. `suggest_fix` MCP Tool

**Purpose:** AI agents can request fix suggestions for detected vulnerabilities.

```python
# New MCP tool for AI agents
async def suggest_fix(
    vulnerability: Vulnerability,
    strategy: str = "auto"  # "parameterize", "escape", "validate", "auto"
) -> FixSuggestion:
    """Generate a verified fix for a security vulnerability."""
    return FixSuggestion(
        vulnerability_id="SQL_INJECTION_L42",
        strategy_used="parameterize",
        original_code='query = f"SELECT * FROM users WHERE id={user_id}"',
        fixed_code='query = "SELECT * FROM users WHERE id=?"\ncursor.execute(query, (user_id,))',
        diff="@@ -42 +42,2 @@\n-query = f\"SELECT...\"\n+query = \"SELECT...\"\n+cursor.execute(...)",
        verification_status="BEHAVIOR_PRESERVED",
        confidence=0.98
    )
```

#### 3. `apply_verified_fix` MCP Tool

**Purpose:** AI agents can apply fixes only after verification passes.

```python
# New MCP tool for AI agents
async def apply_verified_fix(
    file_path: str,
    fix: FixSuggestion,
    require_verification: bool = True
) -> ApplyResult:
    """Apply a fix only if behavior verification passes."""
    # 1. Re-verify the fix
    # 2. Apply if safe
    # 3. Return result with before/after
    return ApplyResult(
        success=True,
        file_modified=file_path,
        lines_changed=[42, 43],
        backup_created=True,
        can_rollback=True
    )
```

### AI Agent Workflow

```
┌─────────────────────────────────────────────────────────────┐
│              AI AGENT VERIFICATION WORKFLOW                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. DETECT                                                   │
│     └── security_scan(file) -> vulnerability at line 42     │
│                                                              │
│  2. GET FIX                                                  │
│     └── suggest_fix(vuln) -> FixSuggestion with diff        │
│                                                              │
│  3. VERIFY                                                   │
│     └── verify_behavior(original, fixed)                    │
│         └── Returns: is_safe=True, confidence=0.95          │
│                                                              │
│  4. APPLY (only if verified)                                 │
│     └── apply_verified_fix(file, fix)                       │
│         └── Creates backup, applies change                  │
│                                                              │
│  5. CONFIRM                                                  │
│     └── security_scan(file) -> 0 vulnerabilities            │
│                                                              │
│  SAFETY GUARANTEES:                                          │
│  - Never applies unverified changes                          │
│  - Always creates backup before modification                 │
│  - Rollback available if issues discovered later             │
│  - Human can review verification results                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Acceptance Criteria Checklist

v2.1.0 Release Criteria:

[ ] verify_behavior: Detects signature changes (P0)
[ ] verify_behavior: Detects return type changes (P0)
[ ] verify_behavior: Detects semantic behavior changes (P0)
[ ] verify_behavior: Returns confidence score (P0)
[ ] verify_behavior: Works for Python/TS/JS/Java (P0)

[ ] suggest_fix: Generates SQL injection fix (parameterize) (P0)
[ ] suggest_fix: Generates XSS fix (escape) (P0)
[ ] suggest_fix: Generates command injection fix (subprocess.run) (P0)
[ ] suggest_fix: Returns unified diff format (P0)
[ ] suggest_fix: Includes verification status (P0)

[ ] apply_verified_fix: Requires verification pass (P0)
[ ] apply_verified_fix: Creates backup before change (P0)
[ ] apply_verified_fix: Supports rollback (P0)
[ ] apply_verified_fix: Returns lines changed (P0)

[ ] Batch verification for multi-file refactors (P1)
[ ] Integration with existing simulate_refactor (P1)

[ ] All new MCP tools registered and documented (Gate)
[ ] All tests passing (Gate)
[ ] Code coverage >= 95% (Gate)
[ ] No regressions in polyglot detections (Gate)

---

## Risk Register

| ID  | Risk                                 | Probability | Impact   | Mitigation                    | Owner |
| --- | ------------------------------------ | ----------- | -------- | ----------------------------- | ----- |
| R1  | Cross-file taint too complex         | High        | High     | Start single-hop, iterate     | TBD   |
| R2  | TypeScript AST differs significantly | Medium      | High     | Use tree-sitter, proven       | TBD   |
| R3  | AI verification gives false confidence | High      | Critical | Conservative confidence scores | TBD   |
| R4  | MCP protocol changes break compatibility | Low      | High     | Pin MCP version, abstract layer | TBD   |
| R5  | Performance degrades at scale        | Medium      | High     | Benchmark at 100k LOC         | TBD   |
| R6  | False positive rate too high         | Medium      | High     | Tune patterns, add sanitizers | TBD   |

---

## Success Metrics

### Quality Gates (All Releases)

| Metric | Threshold | Enforcement |
|--------|-----------|-------------|
| Test Pass Rate | 100% | CI blocks merge |
| Code Coverage | >= 95% | CI blocks merge |
| Ruff Lint | 0 errors | CI blocks merge |
| Black Format | Pass | CI blocks merge |
| Security Scan | 0 new vulns | CI blocks merge |

### Release-Specific KPIs

| Version | KPI | Target |
|---------|-----|--------|
| v1.3.0 | Detection coverage | 95%+ vulnerability types |
| v1.3.0 | extract_code success rate | 100% for valid paths |
| v1.4.0 | New MCP tools functional | get_file_context, get_symbol_references |
| v1.4.0 | XXE/SSTI false negative rate | 0% |
| v1.5.0 | Project map accuracy | Correctly identifies 95%+ of modules |
| v1.5.0 | CVE scan accuracy | 95%+ vs safety-db |
| v2.0.0 | TypeScript extraction parity | Match Python extract_code |
| v2.0.0 | Polyglot security scan | Same detection rate as Python |
| v2.1.0 | Behavior verification accuracy | 95%+ correct verdicts |
| v2.1.0 | Fix suggestion acceptance | 80%+ fixes are valid |

---

## Contributing

### How to Contribute to This Roadmap

1. **Feature Requests:** Open GitHub issue with `[ROADMAP]` prefix
2. **Priority Disputes:** Comment on existing issues with rationale
3. **Implementation:** Claim a feature by commenting "I'll take this"

### Development Workflow

```bash
# 1. Clone and setup
git clone https://github.com/tescolopio/code-scalpel.git
cd code-scalpel
pip install -e ".[dev]"

# 2. Create feature branch
git checkout -b feature/v1.3.0-nosql-injection

# 3. Write failing tests FIRST (TDD)
pytest tests/test_nosql_injection.py  # Should fail

# 4. Implement feature
# Edit src/code_scalpel/...

# 5. Verify
pytest tests/  # All pass
ruff check src/
black --check src/

# 6. Submit PR
git push origin feature/v1.3.0-nosql-injection
# Open PR against main
```

### Code Style Requirements

- **Python 3.9+** minimum
- **Black** formatting (line length 88)
- **Ruff** linting (all rules enabled)
- **Type hints** required for all public functions
- **Docstrings** required for all public classes/functions

---

## Appendix A: Competitor Analysis

| Feature              | Code Scalpel (v2.1.0) | Semgrep | CodeQL | Snyk | Bandit |
|----------------------|-----------------------|---------|--------|------|--------|
| Python security      | ✅                    | ✅      | ✅     | ✅   | ✅     |
| TypeScript security  | ✅                    | ✅      | ✅     | ✅   | ❌     |
| Cross-file taint     | ✅                    | ❌      | ✅     | ❌   | ❌     |
| MCP server for AI    | ✅                    | ❌      | ❌     | ❌   | ❌     |
| Surgical extraction  | ✅                    | ❌      | ❌     | ❌   | ❌     |
| AI-verified fixes    | ✅                    | ❌      | ❌     | ❌   | ❌     |
| Symbolic execution   | ✅                    | ❌      | ❌     | ❌   | ❌     |
| Test generation      | ✅                    | ❌      | ❌     | ❌   | ❌     |
| Open source          | ✅                    | ✅      | ❌     | ❌   | ✅     |
| IDE plugins          | Community             | ✅      | ✅     | ✅   | ❌     |

**Unique Differentiation:** The only tool purpose-built for AI agents to perform surgical code operations without hallucination. Combines precise extraction, symbolic execution, and behavior verification in an MCP-native architecture.

---

## Appendix B: Glossary

| Term                   | Definition                                                            |
| ---------------------- | --------------------------------------------------------------------- |
| **Taint Tracking**     | Tracking data flow from untrusted sources to dangerous sinks          |
| **PDG**                | Program Dependence Graph - represents data/control dependencies       |
| **Symbolic Execution** | Executing code with symbolic values to explore all paths              |
| **MCP**                | Model Context Protocol - Anthropic's standard for AI tool integration |
| **XXE**                | XML External Entity - attack injecting external entities in XML       |
| **SSTI**               | Server-Side Template Injection - code injection via templates         |
| **OODA Loop**          | Observe-Orient-Decide-Act - decision cycle for autonomous agents      |

---

## Document History

| Version | Date       | Author  | Changes                                           |
| ------- | ---------- | ------- | ------------------------------------------------- |
| 1.0     | 2025-12-12 | Copilot | Initial roadmap based on external tester feedback |

---

_This is a living document. Updates will be committed as priorities evolve._

**Questions?** Open a GitHub issue or contact the maintainers.
