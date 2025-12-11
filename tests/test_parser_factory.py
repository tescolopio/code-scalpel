import pytest
from code_scalpel.code_parser.factory import ParserFactory
from code_scalpel.code_parser.interface import Language, IParser
from code_scalpel.code_parser.python_parser import PythonParser


class TestParserFactory:
    def test_detect_language(self):
        assert ParserFactory.detect_language("test.py") == Language.PYTHON
        assert ParserFactory.detect_language("app.js") == Language.JAVASCRIPT
        assert ParserFactory.detect_language("main.java") == Language.JAVA
        assert ParserFactory.detect_language("unknown.xyz") == Language.UNKNOWN

    def test_get_python_parser(self):
        parser = ParserFactory.get_parser(Language.PYTHON)
        assert isinstance(parser, PythonParser)
        assert isinstance(parser, IParser)

    def test_get_unsupported_parser(self):
        with pytest.raises(ValueError, match="No parser registered"):
            ParserFactory.get_parser(Language.JAVASCRIPT)  # Not implemented yet


class TestPythonParser:
    def test_parse_valid_code(self):
        parser = PythonParser()
        code = "def hello(): return 'world'"
        result = parser.parse(code)
        assert result.language == Language.PYTHON
        assert result.ast is not None
        assert result.metrics["complexity"] >= 1
        assert not result.errors

    def test_parse_syntax_error(self):
        parser = PythonParser()
        code = "def broken("
        result = parser.parse(code)
        assert result.ast is None
        assert len(result.errors) > 0
        assert result.errors[0]["type"] == "SyntaxError"

    def test_get_functions(self):
        parser = PythonParser()
        code = """
def func1(): pass
class MyClass:
    def method(self): pass
"""
        result = parser.parse(code)
        funcs = parser.get_functions(result.ast)
        assert "func1" in funcs
        assert "method" in funcs
