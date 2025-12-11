# Code Scalpel v1.2.0 Release Notes

**Release Date:** 2025-01-14

## Headline: Surgical LLM Interaction Toolkit

This release introduces **Surgical Tools** - a new category of features designed for **token-efficient code extraction and modification**. AI agents can now read and write code with up to **99% fewer tokens** compared to traditional approaches.

---

## New Features

### 1. Token-Efficient Extraction (`extract_code` tool)

The `extract_code` MCP tool now accepts `file_path` instead of requiring the full code string:

```json
{
  "file_path": "/path/to/module.py",
  "name": "UserService",
  "extraction_type": "class"
}
```

**Token Savings:**
| Approach | Tokens to Agent | Tokens from Agent |
|----------|-----------------|-------------------|
| Traditional (full file) | ~10,000 | ~10,000 |
| Surgical (file_path) | ~50 | ~200 |
| **Savings** | **99.5%** | **98%** |

### 2. Cross-File Dependency Resolution

Enable `include_cross_file_deps: true` to automatically resolve imports:

```json
{
  "file_path": "/path/to/service.py",
  "name": "process_order",
  "extraction_type": "function",
  "include_cross_file_deps": true
}
```

Returns:
- Primary symbol code
- All imported dependencies from project files
- Full context as combined code string

### 3. Surgical Modification (`update_symbol` tool)

New MCP tool for precision code modification with validation:

```json
{
  "file_path": "/path/to/module.py",
  "target_name": "calculate_total",
  "symbol_type": "function",
  "new_code": "def calculate_total(items):\n    return sum(i.price for i in items)"
}
```

Features:
- Syntax validation before save
- Automatic backup creation
- Class method support (`class_name` parameter)

### 4. Project Crawler (`crawl_project` tool)

Discover project structure without reading files:

```json
{
  "root_path": "/path/to/project",
  "include_patterns": ["*.py"],
  "max_depth": 3
}
```

Returns file tree and module information for navigation.

---

## New Modules

### SurgicalExtractor (`code_scalpel.surgical_extractor`)

```python
from code_scalpel import SurgicalExtractor

# From file path (recommended)
extractor = SurgicalExtractor.from_file("/path/to/module.py")
result = extractor.get_function("process_data")

# Cross-file resolution
resolution = SurgicalExtractor.resolve_cross_file_dependencies(
    file_path="/path/to/service.py",
    symbol_name="UserService",
    project_root="/path/to/project"
)
```

### SurgicalPatcher (`code_scalpel.surgical_patcher`)

```python
from code_scalpel import SurgicalPatcher

patcher = SurgicalPatcher("/path/to/module.py")
patcher.update_function("validate", "def validate(x): return x > 0")
patcher.save()  # Creates backup automatically
```

---

## MCP Server Updates

**Total Tools: 8** (up from 5)

| Tool | Status |
|------|--------|
| `analyze_code` | Stable |
| `security_scan` | Stable |
| `symbolic_execute` | Beta |
| `generate_unit_tests` | Stable |
| `simulate_refactor` | Stable |
| `extract_code` | **NEW** - Token-efficient extraction |
| `update_symbol` | **NEW** - Surgical modification |
| `crawl_project` | **NEW** - Project discovery |

---

## Code Quality Improvements

### extract_code Refactoring
- **Lines of code:** 279 → 118 (58% reduction)
- **Cyclomatic complexity:** 34 branches → 11 branches (68% reduction)
- Extracted 6 focused helper functions

### Test Coverage
- **Total tests:** 1,597 (up from 1,016)
- **Symbolic Execution:** 100% coverage (all 9 modules)
- **New test files:**
  - `test_surgical_extractor.py` (28 tests)
  - `test_surgical_patcher.py` (28 tests)
  - `test_cross_file_resolution.py` (18 tests)
  - `test_pdg_transformer.py` (11 tests)

---

## Documentation Updates

- **README.md:** Updated stats, new tools table
- **docs/api_reference.md:** Full MCP API documentation
- **docs/agent_integration.md:** NEW - Complete integration guide
- **docs/examples.md:** NEW - Practical examples
- **docs/COMPREHENSIVE_GUIDE.md:** New Surgical Tools section

---

## Breaking Changes

None. All existing APIs remain backward compatible.

---

## Upgrade Guide

```bash
pip install --upgrade code-scalpel
```

Or with latest from source:
```bash
pip install git+https://github.com/your-org/code-scalpel.git@v1.2.0
```

---

## What's Next (v1.3.0 Roadmap)

- Float/List/Dict support in symbolic execution
- Semantic diff for update_symbol
- Multi-language support (Java, JavaScript) for surgical tools
- VS Code extension integration

---

## Contributors

- Lead development and architecture
- Self-audit using Code Scalpel's own analysis tools

---

**Full Changelog:** v1.1.1...v1.2.0
