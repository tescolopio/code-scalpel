# Release Notes v1.3.0 - "Hardening"

**Release Date:** December 12, 2025  
**Codename:** Hardening  
**Status:** Stable

---

## Overview

v1.3.0 focuses on hardening the security analysis capabilities and improving the reliability of MCP tools for AI agents. This release adds detection for NoSQL injection, LDAP injection, and hardcoded secrets, while achieving 95%+ test coverage across all surgical tools.

---

## New Features

### Secret Detection (30+ Patterns)

Comprehensive detection of hardcoded secrets in code:

| Category | Patterns | Examples |
|----------|----------|----------|
| AWS | Access keys, Secret keys | `AKIA...`, `aws_secret_access_key=` |
| GitHub | Personal tokens, OAuth tokens | `ghp_*`, `gho_*`, `ghu_*` |
| Stripe | Live and test keys | `sk_live_*`, `sk_test_*` |
| Private Keys | RSA, EC, DSA, OpenSSH | `-----BEGIN PRIVATE KEY-----` |
| Generic | Passwords, API keys | `password=`, `api_key=`, `secret=` |

**Smart Filtering:** Ignores placeholder values like `your_api_key_here`, `TODO`, `CHANGEME`.

### NoSQL Injection Detection (CWE-943)

Detects tainted input flowing to MongoDB operations:

- `find()`, `find_one()`
- `aggregate()`
- `update_one()`, `update_many()`
- `delete_one()`, `delete_many()`
- `insert_one()`, `insert_many()`

Supports both PyMongo and Motor (async) drivers.

### LDAP Injection Detection (CWE-90)

Detects tainted input in LDAP operations:

- `search()`, `search_s()`, `search_ext()`
- `bind()`, `simple_bind_s()`
- `modify()`, `modify_s()`
- `add()`, `delete()`

Supports python-ldap and ldap3 libraries.

### Float Type Inference

Symbolic execution now correctly infers and handles float types:

- `InferredType.FLOAT` added to type system
- Z3 `RealSort` used for float symbolic variables
- Test generation produces correct float test values

### Path Resolution Improvements

`extract_code` MCP tool now handles:

- Relative paths from project root
- Absolute paths
- Clear error messages for missing files

---

## Improvements

### Test Coverage

| Module | Before | After |
|--------|--------|-------|
| SurgicalExtractor | 94% | 95% |
| SurgicalPatcher | 96% | 96% |
| Overall Surgical Tools | 81-87% | 95%+ |

### MCP Tool Enhancements

- `analyze_code` now returns `FunctionInfo` and `ClassInfo` models with line numbers
- All vulnerability reports include exact line numbers
- Better error messages for invalid inputs

---

## Technical Details

### New Taint Sinks

```python
# [20251212_SECURITY] NoSQL injection sinks
NOSQL_SINKS = {
    "find", "find_one", "aggregate",
    "update_one", "update_many",
    "delete_one", "delete_many",
    "insert_one", "insert_many"
}

# [20251212_SECURITY] LDAP injection sinks
LDAP_SINKS = {
    "search", "search_s", "search_ext",
    "bind", "simple_bind_s",
    "modify", "modify_s", "add", "delete"
}
```

### Secret Detection Patterns

30+ regex patterns covering:
- Cloud provider credentials (AWS, GCP, Azure)
- API tokens (GitHub, Stripe, Slack)
- Private keys (RSA, EC, DSA, OpenSSH)
- Generic secrets with smart placeholder filtering

---

## Breaking Changes

None. This release is fully backward compatible.

---

## Migration Guide

No migration required. Simply upgrade:

```bash
pip install --upgrade code-scalpel
```

---

## Test Results

```
1,669 tests passed
0 failed
0 skipped
Coverage: 95%+
```

---

## Contributors

- Development: GitHub Copilot (Claude Opus 4.5)
- Architecture: 3D Tech Solutions LLC
- External Validation: Independent security team (9.5/10 score)

---

## What's Next

See [DEVELOPMENT_ROADMAP.md](../DEVELOPMENT_ROADMAP.md) for upcoming releases:

- **v1.4.0 "Context"**: New MCP tools for context gathering (get_file_context, get_symbol_references)
- **v1.5.0 "Project Intelligence"**: Project-wide analysis tools (get_project_map, get_call_graph)
