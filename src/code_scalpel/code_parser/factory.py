from typing import Dict, Type, Optional
from .interface import IParser, Language
from .python_parser import PythonParser

class ParserFactory:
    """Factory for creating language-specific parsers."""
    
    _parsers: Dict[Language, Type[IParser]] = {
        Language.PYTHON: PythonParser,
    }

    @classmethod
    def get_parser(cls, language: Language) -> IParser:
        """Get a parser instance for the specified language."""
        parser_cls = cls._parsers.get(language)
        if not parser_cls:
            raise ValueError(f"No parser registered for language: {language}")
        return parser_cls()

    @classmethod
    def register_parser(cls, language: Language, parser_cls: Type[IParser]):
        """Register a new parser."""
        cls._parsers[language] = parser_cls

    @staticmethod
    def detect_language(filename: str) -> Language:
        """Detect language from filename extension."""
        ext = filename.split('.')[-1].lower() if '.' in filename else ""
        
        mapping = {
            "py": Language.PYTHON,
            "pyi": Language.PYTHON,
            "js": Language.JAVASCRIPT,
            "jsx": Language.JAVASCRIPT,
            "ts": Language.TYPESCRIPT,
            "tsx": Language.TYPESCRIPT,
            "java": Language.JAVA,
            "cpp": Language.CPP,
            "cc": Language.CPP,
            "h": Language.CPP,
        }
        
        return mapping.get(ext, Language.UNKNOWN)
