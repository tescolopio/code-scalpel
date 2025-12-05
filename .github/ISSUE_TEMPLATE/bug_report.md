---
name: 🐛 Bug Report
about: Something isn't working as expected
title: "[BUG] "
labels: bug, needs-triage
assignees: ''
---

## 🐛 Bug Description

<!--
A clear and concise description of what the bug is.
Bad: "It doesn't work"
Good: "The `analyze` command crashes when given a file with Unicode characters"
-->

## 📋 Environment

<!--
Fill in ALL of these. We cannot reproduce your bug without this information.
-->

| Item | Value |
|------|-------|
| **OS** | <!-- e.g., Ubuntu 22.04, Windows 11, macOS 14.1 --> |
| **Python Version** | <!-- Run: python --version --> |
| **Code Scalpel Version** | <!-- Run: code-scalpel version --> |
| **Installation Method** | <!-- pip, pip install -e ., conda, etc. --> |

## 🔬 Reproduction Script

<!--
This is the MOST IMPORTANT section.
Provide the MINIMAL code that reproduces the bug.
If we cannot reproduce it, we cannot fix it.
-->

```python
# Paste your minimal reproduction script here
from code_scalpel import CodeAnalyzer

analyzer = CodeAnalyzer()
# ... steps that cause the bug
```

**OR** if using CLI:

```bash
# Paste the exact command you ran
code-scalpel analyze --code "your code here"
```

## 💥 Actual Behavior

<!--
What actually happened? Paste the full error message and traceback.
-->

```
Paste error output here
```

## ✅ Expected Behavior

<!--
What did you expect to happen?
-->

## 📎 Additional Context

<!--
Optional: Any other context about the problem.
- Did it work before? (which version?)
- Does it work with different input?
- Screenshots if relevant
-->

## ✔️ Checklist

- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I have provided a minimal reproduction script
- [ ] I have included the full error traceback
- [ ] I have filled in the Environment table completely
