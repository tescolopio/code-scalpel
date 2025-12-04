# Release Protocol

**Version:** 1.0  
**Last Updated:** 2024-12-04  
**Status:** Gate 0 ✅ PASSED

---

## Philosophy

> A Roadmap says "We will do X in Q4."  
> A Release Protocol says "We will not ship X until it passes Gate Y."

We do not ship hope. We ship verified artifacts.

---

## Gate Status Overview

| Gate | Name | Status | Blocking |
|------|------|--------|----------|
| 0 | Security Gate | ✅ PASSED | - |
| 1 | Artifact Gate | 🟡 PENDING | - |
| 2 | Dress Rehearsal | ⚪ NOT STARTED | Gate 1 |
| 3 | Public Debut | ⚪ NOT STARTED | Gate 2 |
| 4 | Redemption | 🧪 EXPERIMENTAL | Gate 3 |

---

## 🛡️ Gate 0: Security Gate

**Status: ✅ PASSED (2024-12-04)**

### Checklist

- [x] **Network Binding Audit**
  - Default bind address changed from `0.0.0.0` to `127.0.0.1`
  - Users must explicitly use `--host 0.0.0.0` for network access
  - Files modified: `mcp_server.py`, `cli.py`

- [x] **Path Traversal Test**
  - MCP server does not read files from disk
  - Only accepts `code` as a string in JSON body
  - Tested with: `../../../../etc/passwd`, `file:///etc/passwd`
  - Result: No file contents leaked

- [x] **Code Execution Test**
  - Dangerous code is ANALYZED, not EXECUTED
  - Tested with: `os.system()`, `eval()`, `exec()`
  - Result: No side effects, security issues detected

- [x] **Dependency Audit**
  - Ran `pip-audit`
  - Created `requirements-secure.txt` with pinned, patched versions
  - Critical fixes: `werkzeug>=3.1.4`, `jinja2>=3.1.6`, `urllib3>=2.5.0`

### Evidence

```
$ python scripts/simulate_mcp_client.py --port 8098
✅ SECURITY: No path traversal (21ms)
   No path traversal detected
✅ SECURITY: No code execution (3ms)
   Code analyzed (not executed), security issues detected
🎉 ALL TESTS PASSED (11/11)
```

---

## 📦 Gate 1: Artifact Gate

**Status: 🟡 PENDING**

### Checklist

- [ ] **Install `check-manifest`**
  ```bash
  pip install check-manifest
  ```

- [ ] **Create/Update MANIFEST.in**
  - Include: `README.md`, `LICENSE`, `py.typed`
  - Exclude: `tests/`, `scripts/`, `docs/`, `.github/`

- [ ] **Build Verification**
  ```bash
  python -m build
  tar -tzf dist/code_scalpel-0.1.0.tar.gz | head -20
  unzip -l dist/code_scalpel-0.1.0-py3-none-any.whl | head -20
  ```

- [ ] **Verify package contents manually**
  - No test files in wheel
  - No secrets or credentials
  - README and LICENSE present

---

## 🎭 Gate 2: Dress Rehearsal (TestPyPI)

**Status: ⚪ NOT STARTED**
**Blocked by:** Gate 1

### Checklist

- [ ] **Configure TestPyPI credentials**
  ```bash
  # In ~/.pypirc or via environment
  TWINE_USERNAME=__token__
  TWINE_PASSWORD=pypi-...
  ```

- [ ] **Upload to TestPyPI**
  ```bash
  twine upload --repository testpypi dist/*
  ```

- [ ] **The "Stranger" Test**
  ```bash
  # Fresh environment, no local code
  python -m venv /tmp/stranger-test
  source /tmp/stranger-test/bin/activate
  pip install -i https://test.pypi.org/simple/ code-scalpel
  
  # Verify
  python -c "from code_scalpel import CodeAnalyzer; print('OK')"
  code-scalpel --help
  code-scalpel version
  code-scalpel analyze --code "def f(): pass"
  ```

- [ ] **Document any dependency issues**
  - TestPyPI may not have all dependencies
  - Use `--extra-index-url https://pypi.org/simple/` if needed

---

## 🚀 Gate 3: Public Debut (v0.1.0)

**Status: ⚪ NOT STARTED**
**Blocked by:** Gate 2

### Scope

**Included:**
- AST Analysis (94% coverage)
- PDG Analysis (86% coverage)
- MCP Server (65% coverage, integration tested)
- CLI (76% coverage)
- Integrations: Autogen, CrewAI (67-70% coverage)

**Excluded (Quarantined):**
- Symbolic Execution (experimental, crashes on use)

### Checklist

- [ ] **Final test run**
  ```bash
  python -m pytest tests/ --cov=src/code_scalpel
  # Must pass: 180+ tests, 37%+ coverage
  ```

- [ ] **Version bump to `0.1.0`**
  - Update `pyproject.toml`
  - Update `__version__` if exists

- [ ] **Upload to PyPI**
  ```bash
  twine upload dist/*
  ```

- [ ] **Create GitHub Release**
  - Tag: `v0.1.0`
  - Title: "Code Scalpel v0.1.0 - Initial Release"
  - Body: Release notes with features, known limitations

- [ ] **Verify installation**
  ```bash
  pip install code-scalpel
  code-scalpel version
  ```

---

## 🔮 Gate 4: Redemption (v0.2.0)

**Status: 🧪 EXPERIMENTAL**
**Blocked by:** Gate 3

### Goal

Re-enable Symbolic Execution as a production feature.

### Requirements

- [ ] Implement `SymbolicExecutionEngine._infer_type()`
- [ ] Implement `ConstraintSolver.solve()`
- [ ] Integrate Z3 solver properly
- [ ] Achieve 80% coverage on `symbolic_execution_tools/`
- [ ] Remove `UserWarning` from `__init__.py`
- [ ] Update README to move from "Experimental" to "Features"

### Evidence Required

```bash
python -c "
from code_scalpel.symbolic_execution_tools import SymbolicExecutionEngine
from code_scalpel.symbolic_execution_tools import ConstraintSolver

solver = ConstraintSolver()
engine = SymbolicExecutionEngine(solver)
result = engine.execute('x = 1 + 2')
print('Paths explored:', len(result.paths))
"
# Must complete without error
```

---

## Audit Trail

| Date | Gate | Action | Result |
|------|------|--------|--------|
| 2024-12-04 | 0 | Security audit | ✅ PASSED |
| 2024-12-04 | 0 | Fixed 0.0.0.0 → 127.0.0.1 | ✅ Fixed |
| 2024-12-04 | 0 | Path traversal test | ✅ No vulnerability |
| 2024-12-04 | 0 | Code execution test | ✅ No vulnerability |
| 2024-12-04 | 0 | pip-audit | ⚠️ 47 CVEs in system, mitigated with requirements-secure.txt |
