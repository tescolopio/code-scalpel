"""
Tests for framework-specific security sink detection.

This module tests detection of SQL injection and XSS vulnerabilities
in Django, SQLAlchemy, and Flask/Jinja2 code patterns.
"""


class TestDjangoSQLInjection:
    """Test detection of Django-specific SQL injection patterns."""

    def test_rawsql_with_user_input(self):
        """Detect SQL injection via Django RawSQL."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
from django.db.models.expressions import RawSQL
user_input = request.GET.get("order")
queryset = MyModel.objects.annotate(val=RawSQL(user_input, []))
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities
        vuln_types = [v.vulnerability_type for v in result.vulnerabilities]
        assert any("SQL" in vt for vt in vuln_types)

    def test_rawsql_short_import(self):
        """Detect SQL injection via RawSQL short import."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
from django.db.models import RawSQL
search = request.POST.get("search")
qs = Model.objects.annotate(custom=RawSQL(search, []))
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities
        assert any(v.cwe_id == "CWE-89" for v in result.vulnerabilities)

    def test_queryset_extra_with_tainted_data(self):
        """Detect SQL injection via QuerySet.extra()."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
order_by = request.GET.get("sort")
qs = extra(order_by)
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities

    def test_safe_rawsql_with_params(self):
        """Safe RawSQL with parameterized query should not flag if no taint."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
from django.db.models import RawSQL
# Safe: hardcoded SQL, no user input
queryset = MyModel.objects.annotate(val=RawSQL("price * %s", [1.05]))
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        # No tainted data reaches the sink
        assert not result.has_vulnerabilities


class TestSQLAlchemyInjection:
    """Test detection of SQLAlchemy-specific SQL injection patterns."""

    def test_text_with_user_input(self):
        """Detect SQL injection via sqlalchemy.text()."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
from sqlalchemy import text
user_query = request.args.get("query")
result = session.execute(text(user_query))
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities
        vuln_types = [v.vulnerability_type for v in result.vulnerabilities]
        assert any("SQL" in vt for vt in vuln_types)

    def test_text_full_path(self):
        """Detect SQL injection via full sqlalchemy.text path."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
import sqlalchemy
query_str = input("Enter query: ")
stmt = sqlalchemy.text(query_str)
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities
        assert any(v.cwe_id == "CWE-89" for v in result.vulnerabilities)

    def test_text_expression_module(self):
        """Detect SQL injection via sqlalchemy.sql.expression.text."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
from sqlalchemy.sql.expression import text
search = request.form["search"]
query = text(search)
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities

    def test_safe_text_with_bindparams(self):
        """Safe sqlalchemy.text with bind parameters should not flag."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
from sqlalchemy import text
# Safe: hardcoded SQL template
stmt = text("SELECT * FROM users WHERE id = :id")
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        # No tainted data
        assert not result.has_vulnerabilities


class TestFlaskJinja2XSS:
    """Test detection of Flask/Jinja2 XSS patterns."""

    def test_flask_markup_with_user_input(self):
        """Detect XSS via flask.Markup bypassing auto-escape."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
from flask import Markup
user_html = request.form.get("content")
safe_html = Markup(user_html)  # DANGEROUS: bypasses auto-escaping
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities
        vuln_types = [v.vulnerability_type for v in result.vulnerabilities]
        assert any("XSS" in vt or "Cross-Site" in vt for vt in vuln_types)

    def test_markupsafe_markup_with_tainted_data(self):
        """Detect XSS via markupsafe.Markup."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
from markupsafe import Markup
comment = request.args.get("comment")
rendered = Markup(comment)
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities
        assert any(v.cwe_id == "CWE-79" for v in result.vulnerabilities)

    def test_markup_short_import(self):
        """Detect XSS via Markup short import."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
from markupsafe import Markup
user_input = input("Enter HTML: ")
html = Markup(user_input)
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities

    def test_safe_markup_with_escaped_content(self):
        """Safe Markup with pre-escaped content should not flag if no taint."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
from markupsafe import Markup
# Safe: hardcoded HTML, no user input
header = Markup("<h1>Welcome</h1>")
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        # No tainted data reaches the sink
        assert not result.has_vulnerabilities


class TestCombinedFrameworkPatterns:
    """Test detection across multiple framework patterns."""

    def test_django_and_flask_vulnerabilities(self):
        """Detect both SQL injection and XSS in same codebase."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
from django.db.models import RawSQL
from flask import Markup

search = request.GET.get("q")
comment = request.POST.get("comment")

# SQL Injection
results = Model.objects.annotate(score=RawSQL(search, []))

# XSS
rendered = Markup(comment)
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities
        assert len(result.vulnerabilities) >= 2

        cwe_ids = {v.cwe_id for v in result.vulnerabilities}
        assert "CWE-89" in cwe_ids  # SQL Injection
        assert "CWE-79" in cwe_ids  # XSS


class TestTaintTrackerFrameworkIntegration:
    """Test TaintTracker with framework sink patterns."""

    def test_sink_patterns_include_django(self):
        """Verify Django sinks are registered."""
        from code_scalpel.symbolic_execution_tools.taint_tracker import SINK_PATTERNS

        assert "RawSQL" in SINK_PATTERNS
        assert "django.db.models.expressions.RawSQL" in SINK_PATTERNS

    def test_sink_patterns_include_sqlalchemy(self):
        """Verify SQLAlchemy sinks are registered."""
        from code_scalpel.symbolic_execution_tools.taint_tracker import SINK_PATTERNS

        assert "text" in SINK_PATTERNS
        assert "sqlalchemy.text" in SINK_PATTERNS

    def test_sink_patterns_include_flask(self):
        """Verify Flask/Jinja2 sinks are registered."""
        from code_scalpel.symbolic_execution_tools.taint_tracker import SINK_PATTERNS

        assert "Markup" in SINK_PATTERNS
        assert "flask.Markup" in SINK_PATTERNS
        assert "markupsafe.Markup" in SINK_PATTERNS
