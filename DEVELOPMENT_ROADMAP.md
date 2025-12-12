# Code Scalpel Development Roadmap

**Document Version:** 1.0  
**Last Updated:** December 12, 2025  
**Current Release:** v1.2.3 (Stable)  
**Maintainer:** 3D Tech Solutions LLC

---

## Executive Summary

Code Scalpel is evolving from a Python-focused security scanner into the **dominant polyglot code analysis platform** for AI-assisted development. This roadmap outlines our development priorities through 2025, incorporating feedback from external security testers who validated our v1.2.3 release with a **9.5/10 rating**.

### Current State (v1.2.3)

| Metric | Value | Status |
|--------|-------|--------|
| Vulnerability Detection Rate | 100% (12/12 types) | ✅ Production |
| Test Suite | 1,669 tests passing | ✅ Stable |
| Code Coverage | 95%+ | ✅ Target Met |
| Languages | Python (full), JS/Java (structural) | 🟡 Expanding |
| MCP Tools | 8 tools | ✅ Stable |

### Target State (Q4 2025)

| Metric | Target | Milestone |
|--------|--------|-----------|
| Vulnerability Types | 25+ | v1.4.0 |
| Languages | Python, TypeScript, JavaScript, Java | v2.0.0 |
| Cross-File Analysis | Full taint tracking | v1.5.1 |
| IDE Integration | VS Code extension | v2.1.0 |
| Agentic Capabilities | Auto-fix with verification | v2.1.0 |

---

## Release Timeline

```
2025 Q1                    2025 Q2                    2025 Q3
│                          │                          │
▼                          ▼                          ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ v1.3.0  │  │ v1.4.0  │  │ v1.5.0  │  │ v1.5.1  │  │ v2.0.0  │  │ v2.1.0  │
│ Harden  │→ │ Enter-  │→ │ Depend- │→ │ Cross-  │→ │ Poly-   │→ │ Agentic │
│         │  │ prise   │  │ encies  │  │ File    │  │ glot    │  │         │
│ Jan '25 │  │ Feb '25 │  │ Mar '25 │  │ Apr '25 │  │ Q2 '25  │  │ Q3 '25  │
└─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘
     │            │            │            │            │            │
     │            │            │            │            │            │
   NoSQL       XXE/SSTI     CVE Scan    Cross-File   TypeScript   Auto-Fix
   LDAP        VS Code      Call Graph  Taint        JavaScript   VS Code
   Secrets     Preview      Circular    Import Res   Java Spring  Full IDE
```

---

## v1.3.0 - "Hardening" (January 2025)

### Overview

**Theme:** Stability and Security Coverage  
**Goal:** Fix critical blockers, expand detection to 95%+  
**Effort:** ~10 developer-days  
**Risk Level:** Low (incremental improvements)

### Priorities

| Priority | Feature | Owner | Effort | Dependencies |
|----------|---------|-------|--------|--------------|
| **P0** | Fix `extract_code` file path resolution | TBD | 2 days | None |
| **P0** | Add hardcoded secret detection | TBD | 1 day | None |
| **P0** | Add NoSQL injection (MongoDB) | TBD | 1 day | None |
| **P0** | Add LDAP injection sinks | TBD | 1 day | None |
| **P0** | Surgical tools → 95% coverage | TBD | 3 days | None |
| **P1** | Line numbers in all MCP tools | TBD | 1 day | None |
| **P1** | Improve test generation types | TBD | 2 days | None |

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
- [ ] `extract_code("utils.py", ...)` works from project root
- [ ] `extract_code("src/utils.py", ...)` works with relative paths
- [ ] `extract_code("/absolute/path/utils.py", ...)` works unchanged
- [ ] Clear error message when file truly doesn't exist

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

```
v1.3.0 Release Criteria:

[ ] extract_code works with relative paths (P0)
[ ] extract_code works with absolute paths (P0)
[ ] extract_code provides clear error for missing files (P0)

[ ] Detects AWS access keys (P0)
[ ] Detects AWS secret keys (P0)
[ ] Detects GitHub tokens (ghp_, gho_, ghu_) (P0)
[ ] Detects Stripe keys (sk_live_, sk_test_) (P0)
[ ] Detects private keys (-----BEGIN PRIVATE KEY-----) (P0)
[ ] Detects generic secrets (password=, api_key=) (P0)

[ ] Detects MongoDB find() with tainted input (P0)
[ ] Detects MongoDB aggregate() with tainted input (P0)
[ ] Detects MongoDB update/delete with tainted input (P0)

[ ] Detects LDAP search with tainted filter (P0)
[ ] Detects LDAP bind with tainted credentials (P0)

[ ] SurgicalExtractor coverage ≥ 95% (P0)
[ ] SurgicalPatcher coverage ≥ 95% (P0)

[ ] All MCP tools return line numbers (P1)
[ ] Test generation infers float types correctly (P1)

[ ] All 1,669+ tests passing (Gate)
[ ] No regressions in existing detections (Gate)
[ ] Documentation updated (Gate)
```

---

## v1.4.0 - "Enterprise" (February 2025)

### Overview

**Theme:** Enterprise Security + IDE Integration  
**Goal:** Add critical vulnerability types, ship VS Code preview  
**Effort:** ~15 developer-days  
**Risk Level:** Medium (new architecture for VS Code)

### Priorities

| Priority | Feature | Owner | Effort | Dependencies |
|----------|---------|-------|--------|--------------|
| **P0** | XXE detection | TBD | 2 days | None |
| **P0** | SSTI detection (Jinja2) | TBD | 1 day | None |
| **P0** | VS Code extension (Preview) | TBD | 5 days | None |
| **P1** | JWT vulnerabilities | TBD | 2 days | None |
| **P1** | Mass assignment detection | TBD | 2 days | None |
| **P2** | Open redirect | TBD | 1 day | None |
| **P2** | Header injection | TBD | 1 day | None |

### Technical Specifications

#### 1. XXE Detection (XML External Entity)

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

#### 3. VS Code Extension (Preview)

**Scope (MVP only):**
```
✅ In Scope:
- Right-click file → "Code Scalpel: Scan for Vulnerabilities"
- Results shown in Problems panel
- Diagnostic markers (red squiggles) on vulnerable lines
- Quick-info hover showing vulnerability details
- Command palette: "Code Scalpel: Scan Current File"

❌ Out of Scope (v2.1.0):
- Auto-fix suggestions
- Inline code actions
- Project-wide dashboard
- Settings UI
```

**Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                    VS Code Extension                     │
├─────────────────────────────────────────────────────────┤
│  extension.ts                                            │
│  ├── registerCommand("codescalpel.scanFile")            │
│  ├── DiagnosticCollection for Problems panel            │
│  └── HoverProvider for vulnerability details            │
├─────────────────────────────────────────────────────────┤
│  Code Scalpel MCP Server (localhost:8593)               │
│  └── security_scan tool                                  │
└─────────────────────────────────────────────────────────┘
```

**package.json (extension manifest):**
```json
{
  "name": "code-scalpel",
  "displayName": "Code Scalpel",
  "description": "AI-powered security scanner for Python",
  "version": "0.1.0",
  "engines": { "vscode": "^1.80.0" },
  "categories": ["Linters", "Programming Languages"],
  "activationEvents": ["onLanguage:python"],
  "contributes": {
    "commands": [
      {
        "command": "codescalpel.scanFile",
        "title": "Scan for Vulnerabilities",
        "category": "Code Scalpel"
      }
    ],
    "menus": {
      "editor/context": [
        {
          "command": "codescalpel.scanFile",
          "when": "resourceLangId == python"
        }
      ]
    }
  }
}
```

---

## v1.5.0 - "Dependencies" (March 2025)

### Overview

**Theme:** Dependency Intelligence (Quick Wins)  
**Goal:** High-value, low-effort features  
**Effort:** ~7 developer-days  
**Risk Level:** Low (uses existing infrastructure)

### Priorities

| Priority | Feature | Owner | Effort | Dependencies |
|----------|---------|-------|--------|--------------|
| **P0** | Dependency CVE scan | TBD | 3 days | None |
| **P0** | Call graph visualization | TBD | 2 days | PDG exists |
| **P1** | Circular dependency detection | TBD | 1 day | PDG exists |

### Technical Specifications

#### 1. Dependency CVE Scan

**Data Sources:**
- OSV (Open Source Vulnerabilities) API - Free, comprehensive
- NVD (National Vulnerability Database) - Government source
- PyPI Advisory Database - Python-specific

**Implementation:**
```python
# New MCP tool: scan_dependencies

async def scan_dependencies(requirements_path: str) -> DependencyReport:
    """Scan requirements.txt or pyproject.toml for known CVEs."""
    
    # Parse dependencies
    deps = parse_requirements(requirements_path)
    
    # Query OSV API
    vulnerabilities = []
    for dep in deps:
        osv_results = await query_osv(dep.name, dep.version)
        vulnerabilities.extend(osv_results)
    
    return DependencyReport(
        total_dependencies=len(deps),
        vulnerable_count=len(vulnerabilities),
        vulnerabilities=vulnerabilities,
    )
```

#### 2. Call Graph Visualization

**Output Format:** DOT (Graphviz) or Mermaid

```python
# New MCP tool: visualize_call_graph

def visualize_call_graph(file_path: str, format: str = "mermaid") -> str:
    """Generate call graph visualization from PDG."""
    
    pdg = build_pdg(file_path)
    
    if format == "mermaid":
        return pdg_to_mermaid(pdg)
    elif format == "dot":
        return pdg_to_dot(pdg)
```

**Example Output (Mermaid):**
```mermaid
graph TD
    main --> process_request
    process_request --> validate_input
    process_request --> execute_query
    execute_query --> db.execute
```

---

## v1.5.1 - "CrossFile" (April 2025)

### Overview

**Theme:** Cross-File Taint Tracking  
**Goal:** Track data flow across module boundaries  
**Effort:** ~15 developer-days  
**Risk Level:** High (architectural complexity)

### Priorities

| Priority | Feature | Owner | Effort | Dependencies |
|----------|---------|-------|--------|--------------|
| **P0** | Cross-file taint tracking | TBD | 10 days | Import resolution |
| **P0** | Import resolution | TBD | 5 days | None |

### Architecture

```
Challenge: Track taint across files

File: views.py                    File: models.py
─────────────                     ─────────────
def handle_request(req):          def get_user(user_id):
    user_id = req.args['id']  ──────→  query = f"SELECT * FROM users WHERE id={user_id}"
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

---

## v2.0.0 - "Polyglot" (Q2 2025)

### Overview

**Theme:** TypeScript/JavaScript Support  
**Goal:** Cover 95% of web development  
**Effort:** ~25 developer-days  
**Risk Level:** High (new language architecture)

### Priorities

| Priority | Feature | Owner | Effort | Dependencies |
|----------|---------|-------|--------|--------------|
| **P0** | TypeScript AST support | TBD | 10 days | tree-sitter |
| **P0** | JavaScript security scanning | TBD | 8 days | TS support |
| **P1** | Java Spring security | TBD | 7 days | tree-sitter |
| **P1** | JSX/TSX support | TBD | 3 days | TS support |

### JavaScript/TypeScript Vulnerabilities

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

---

## v2.1.0 - "Agentic" (Q3 2025)

### Overview

**Theme:** AI-Assisted Remediation  
**Goal:** Close the OODA loop with auto-fix  
**Effort:** ~25 developer-days  
**Risk Level:** High (safety-critical)

### Priorities

| Priority | Feature | Owner | Effort | Dependencies |
|----------|---------|-------|--------|--------------|
| **P0** | `auto_fix` MCP tool | TBD | 10 days | simulate_refactor |
| **P0** | VS Code extension (Full) | TBD | 8 days | v1.4.0 extension |
| **P0** | `refactor_safe` tool | TBD | 5 days | simulate_refactor |
| **P1** | CI/CD templates | TBD | 3 days | None |
| **P1** | Framework migration | TBD | 5 days | cross-file |

### Auto-Fix Safety Model

```
┌─────────────────────────────────────────────────────────────┐
│                     AUTO-FIX PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. IDENTIFY                                                 │
│     └── security_scan → vulnerability at line 42            │
│                                                              │
│  2. GENERATE                                                 │
│     └── auto_fix(vuln, strategy="parameterize")             │
│         └── Produces candidate fix (diff)                   │
│                                                              │
│  3. VERIFY                                                   │
│     ├── simulate_refactor(old_code, new_code)               │
│     │   └── Must return: "Behavior preserved"               │
│     ├── Run test suite (if exists)                          │
│     │   └── Must pass all existing tests                    │
│     └── security_scan(new_code)                             │
│         └── Must have 0 new vulnerabilities                 │
│                                                              │
│  4. APPLY                                                    │
│     ├── If CI context: Create PR with diff                  │
│     ├── If IDE context: Show inline suggestion              │
│     └── If CLI context: Apply with --auto-apply flag        │
│                                                              │
│  SAFETY GATES:                                               │
│  ✗ Never auto-apply to main/master branch                   │
│  ✗ Never apply if test suite fails                          │
│  ✗ Never apply if behavior changes detected                 │
│  ✗ Require human approval for production deployments        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Risk Register

| ID | Risk | Probability | Impact | Mitigation | Owner |
|----|------|-------------|--------|------------|-------|
| R1 | Cross-file taint too complex | High | High | Start single-hop, iterate | TBD |
| R2 | TypeScript AST differs significantly | Medium | High | Use tree-sitter, proven | TBD |
| R3 | Auto-fix breaks production code | High | Critical | Mandatory test verification | TBD |
| R4 | VS Code extension scope creep | Medium | Medium | Strict MVP definition | TBD |
| R5 | Performance degrades at scale | Medium | High | Benchmark at 100k LOC | TBD |
| R6 | False positive rate too high | Medium | High | Tune patterns, add sanitizers | TBD |

---

## Success Metrics

### Quality Gates (All Releases)

| Metric | Threshold | Enforcement |
|--------|-----------|-------------|
| Test Pass Rate | 100% | CI blocks merge |
| Code Coverage | ≥ 95% | CI blocks merge |
| Ruff Lint | 0 errors | CI blocks merge |
| Black Format | Pass | CI blocks merge |
| Security Scan | 0 new vulns | CI blocks merge |

### Release-Specific KPIs

| Version | KPI | Target |
|---------|-----|--------|
| v1.3.0 | Detection coverage | 95%+ vulnerability types |
| v1.3.0 | extract_code success rate | 100% for valid paths |
| v1.4.0 | VS Code installs (month 1) | 100+ |
| v1.4.0 | XXE/SSTI false negative rate | 0% |
| v1.5.0 | CVE scan accuracy | 95%+ vs safety-db |
| v2.0.0 | TypeScript parity | Match Python detection |
| v2.1.0 | Auto-fix acceptance rate | 80%+ |

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

| Feature | Code Scalpel (v2.1.0) | Semgrep | CodeQL | Snyk | Bandit |
|---------|----------------------|---------|--------|------|--------|
| Python security | ✅ | ✅ | ✅ | ✅ | ✅ |
| TypeScript security | ✅ | ✅ | ✅ | ✅ | ❌ |
| Cross-file taint | ✅ | ❌ | ✅ | ❌ | ❌ |
| AI agent integration | ✅ MCP | ❌ | ❌ | ❌ | ❌ |
| Auto-fix | ✅ | ✅ | ❌ | ✅ | ❌ |
| Symbolic execution | ✅ | ❌ | ❌ | ❌ | ❌ |
| Test generation | ✅ | ❌ | ❌ | ❌ | ❌ |
| Open source | ✅ | ✅ | ❌ | ❌ | ✅ |
| VS Code extension | ✅ | ✅ | ✅ | ✅ | ❌ |

**Unique Differentiation:** Only tool combining symbolic execution + MCP integration + auto-fix in open source.

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Taint Tracking** | Tracking data flow from untrusted sources to dangerous sinks |
| **PDG** | Program Dependence Graph - represents data/control dependencies |
| **Symbolic Execution** | Executing code with symbolic values to explore all paths |
| **MCP** | Model Context Protocol - Anthropic's standard for AI tool integration |
| **XXE** | XML External Entity - attack injecting external entities in XML |
| **SSTI** | Server-Side Template Injection - code injection via templates |
| **OODA Loop** | Observe-Orient-Decide-Act - decision cycle for autonomous agents |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-12 | Copilot | Initial roadmap based on external tester feedback |

---

*This is a living document. Updates will be committed as priorities evolve.*

**Questions?** Open a GitHub issue or contact the maintainers.
