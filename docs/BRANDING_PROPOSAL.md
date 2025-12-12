# Code Scalpel Branding Proposal

## Status: DRAFT for Review

**Date:** 2025-01-XX  
**Version:** v1.3.0 Target  
**Breaking Change:** Yes (API names)

---

## Current State (v1.2.0)

### MCP Tools (8 tools)
| Current Name | Purpose | Medical Analogy |
|-------------|---------|-----------------|
| `analyze_code` | AST structure analysis | Initial examination |
| `security_scan` | Taint flow analysis | Pathology screening |
| `symbolic_execute` | Path exploration | Diagnostic imaging |
| `generate_unit_tests` | Test case generation | Treatment protocol |
| `simulate_refactor` | Change impact analysis | Surgical simulation |
| `extract_code` | Context extraction | Biopsy |
| `update_symbol` | Code modification | Surgery/Graft |
| `crawl_project` | Project mapping | Full body scan |

### Internal Modules
| Module | Class Names | Status |
|--------|-------------|--------|
| ast_tools | ASTAnalyzer, ASTBuilder | Stable |
| pdg_tools | PDGBuilder, PDGAnalyzer, PDGSlicer | Stable |
| symbolic_execution_tools | SymbolicAnalyzer, ConstraintSolver | Stable |
| security_tools | TaintTracker, SecurityAnalyzer | Stable |
| surgical | SurgicalExtractor, SurgicalPatcher | Beta |

---

## Proposed Branding (v1.3.0)

### Philosophy: Medical Metaphor Consistency

**Tiers:**
1. **Diagnostic (Clinical)** - Read-only analysis, no modifications
2. **Intervention (Surgical)** - Code modifications with safety checks

### MCP Tool Renaming

| Current | Proposed | Tier | Rationale |
|---------|----------|------|-----------|
| `analyze_code` | `examine_structure` | Diagnostic | Initial patient examination |
| `security_scan` | `screen_pathology` | Diagnostic | Cancer screening analogy |
| `symbolic_execute` | `trace_pathways` | Diagnostic | Following blood flow/nerves |
| `generate_unit_tests` | `prescribe_tests` | Diagnostic | Treatment prescription |
| `simulate_refactor` | `simulate_procedure` | Diagnostic | Surgical simulation |
| `extract_code` | `biopsy_code` | Diagnostic | Tissue sample extraction |
| `update_symbol` | `graft_code` | Intervention | Tissue graft/transplant |
| `crawl_project` | `full_body_scan` | Diagnostic | Comprehensive imaging |

### Internal Class Renaming (Optional - Lower Priority)

| Current | Proposed | Notes |
|---------|----------|-------|
| SymbolicAnalyzer | Pathologist | Examines execution paths |
| SurgicalExtractor | Biopsy | Extracts code samples |
| SurgicalPatcher | Surgeon | Performs modifications |
| PDGSlicer | Resector | Removes dependencies |
| TaintTracker | Epidemiologist | Tracks data spread |

---

## Migration Strategy

### Phase 1: Aliases (v1.3.0)
- Add new names as aliases to existing functions
- Deprecation warnings on old names
- Documentation updated to prefer new names

```python
# Example implementation
@mcp.tool(name="biopsy_code")
async def biopsy_code(...):
    """Alias for extract_code with medical branding."""
    return await extract_code(...)

# Deprecation decorator
@deprecated(reason="Use biopsy_code instead", version="1.3.0")
@mcp.tool(name="extract_code")
async def extract_code(...):
    ...
```

### Phase 2: Primary (v1.4.0)
- New names become primary in docs/examples
- Old names still work but show deprecation warnings
- All new tutorials use new names

### Phase 3: Removal (v2.0.0)
- Old names removed
- Clean break for major version

---

## Decision Matrix

### Arguments For Renaming

1. **Memorable:** "biopsy" vs "extract_code" - more distinctive
2. **Consistent:** Full medical metaphor across all tools
3. **Marketing:** Differentiates from generic "code analyzer" tools
4. **Grouping:** Clear Diagnostic vs Intervention tiers

### Arguments Against Renaming

1. **Breaking Change:** Existing users must update
2. **Discoverability:** Generic names easier to find in search
3. **Learning Curve:** New users must learn metaphor
4. **Maintenance:** Two name systems during transition

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| User confusion | Medium | Extensive documentation |
| Ecosystem breakage | Medium | 3-phase migration |
| Search discoverability | Low | Keep generic names in docs |
| International users | Low | Metaphor is universal |

---

## Recommendation

**Option A (Conservative):** Keep current names, enhance documentation with medical metaphors in descriptions only.

**Option B (Moderate):** Implement aliases in v1.3.0, evaluate adoption, decide on deprecation later.

**Option C (Aggressive):** Full rename in v1.3.0 with deprecation warnings.

**Recommendation:** Option B (Moderate)

- Adds branding value without forcing change
- Allows organic adoption metrics
- Preserves backward compatibility
- Can accelerate or slow based on feedback

---

## Implementation Checklist

If approved:
- [ ] Add `@deprecated` decorator utility
- [ ] Create alias functions for all 8 MCP tools
- [ ] Update README with new names prominently
- [ ] Add migration guide to docs
- [ ] Update examples to use new names
- [ ] Add usage telemetry (opt-in) for name preference
- [ ] Create v1.3.0 release notes explaining change

---

## Appendix: Full Name Mapping

### MCP Resources (Keep as-is)
Resources use `scalpel://` URI scheme which already fits the metaphor:
- `scalpel://project/call-graph` - OK
- `scalpel://project/dependencies` - OK
- `scalpel://project/structure` - OK

### CLI Commands (Future consideration)
```bash
# Current
scalpel analyze file.py
scalpel extract file.py::function_name

# Proposed
scalpel examine file.py
scalpel biopsy file.py::function_name
```

---

## Stakeholder Sign-off

- [ ] Lead Architect: _______________
- [ ] Product Owner: _______________
- [ ] Documentation: _______________

---

*This document is a proposal for discussion. No changes will be implemented without explicit approval.*
