import os
import json
import re
from typing import List, Dict

try:
    import tomllib
except ImportError:
    import tomli as tomllib


class DependencyParser:
    """Parses project dependencies from standard configuration files."""

    def __init__(self, root_path: str):
        self.root_path = root_path

    def get_dependencies(self) -> Dict[str, List[Dict[str, str]]]:
        """Returns dependencies grouped by ecosystem."""
        deps = {
            "python": self._parse_python_deps(),
            "javascript": self._parse_javascript_deps(),
        }
        return {k: v for k, v in deps.items() if v}

    def _parse_python_deps(self) -> List[Dict[str, str]]:
        deps = []

        # 1. pyproject.toml (PEP 621 & Poetry)
        pp_path = os.path.join(self.root_path, "pyproject.toml")
        if os.path.exists(pp_path):
            try:
                with open(pp_path, "rb") as f:
                    data = tomllib.load(f)

                # Standard PEP 621
                if "project" in data and "dependencies" in data["project"]:
                    for d in data["project"]["dependencies"]:
                        deps.append(self._parse_pep508(d))

                # Poetry
                if (
                    "tool" in data
                    and "poetry" in data["tool"]
                    and "dependencies" in data["tool"]["poetry"]
                ):
                    for k, v in data["tool"]["poetry"]["dependencies"].items():
                        if k.lower() != "python":
                            deps.append({"name": k, "version": str(v)})
            except Exception:
                pass  # Fail silently, we are scanning

        # 2. requirements.txt
        req_path = os.path.join(self.root_path, "requirements.txt")
        if os.path.exists(req_path):
            try:
                with open(req_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if (
                            line
                            and not line.startswith("#")
                            and not line.startswith("-")
                        ):
                            deps.append(self._parse_pep508(line))
            except Exception:
                pass

        return self._deduplicate(deps)

    def _parse_javascript_deps(self) -> List[Dict[str, str]]:
        deps = []
        pj_path = os.path.join(self.root_path, "package.json")
        if os.path.exists(pj_path):
            try:
                with open(pj_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.get("dependencies", {}).items():
                    deps.append({"name": k, "version": v})
                for k, v in data.get("devDependencies", {}).items():
                    deps.append({"name": k, "version": v, "type": "dev"})
            except Exception:
                pass
        return deps

    def _parse_pep508(self, s: str) -> Dict[str, str]:
        # Basic parsing: "requests>=2.0" -> name="requests", version=">=2.0"
        s = s.split(";")[0].split("#")[0].strip()
        match = re.match(r"^([a-zA-Z0-9_\-\.]+)(.*)$", s)
        if match:
            return {"name": match.group(1), "version": match.group(2).strip() or "*"}
        return {"name": s, "version": "*"}

    def _deduplicate(self, deps: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        unique = []
        for d in deps:
            if d["name"] not in seen:
                seen.add(d["name"])
                unique.append(d)
        return unique
