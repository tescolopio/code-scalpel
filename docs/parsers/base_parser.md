# Base Code Parser

The `base_parser.py` module serves as a foundation for code parsing and analysis across different programming languages. It provides common functionality, abstractions, and utilities that can be inherited and extended by language-specific parsers.

## Features

1. **Language Enum**: Defines an enumeration of supported programming languages, making it easy to specify and identify the language being parsed.

2. **ParseResult**: Represents the result of code parsing, containing the following information:
   - `ast`: The Abstract Syntax Tree (AST) representation of the parsed code.
   - `errors`: A list of parsing errors encountered during the parsing process.
   - `warnings`: A list of warnings or potential issues identified in the code.
   - `tokens`: The token stream generated from the parsed code.
   - `metrics`: A dictionary of various metrics calculated during code analysis.
   - `language`: The programming language of the parsed code.

3. **PreprocessorConfig**: Represents the configuration options for code preprocessing, allowing customization of preprocessing steps such as comment removal, macro expansion, whitespace normalization, and directive handling.

4. **BaseParser**: An abstract base class that defines the common interface and functionality for language-specific parsers. It includes:
   - `parse_code`: An abstract method to be implemented by language-specific parsers, responsible for parsing the code and returning a ParseResult.
   - `preprocess_code`: A method that performs code preprocessing based on the provided PreprocessorConfig, applying language-specific preprocessing steps.
   - `remove_comments`: A static method for removing comments from the code while preserving line numbers.
   - `expand_macros`: A static method for expanding macros in the code.
   - `normalize_whitespace`: A static method for normalizing whitespace in the code.
   - `handle_directives`: A static method for handling preprocessing directives in the code.
   - `calculate_complexity`: A static method for calculating the cyclomatic complexity of code constructs.
   - `format_syntax_error`: A static method for formatting syntax errors with detailed information.

5. **CodeParser**: A class that represents the main code parser with multi-language support. It manages the initialization and coordination of language-specific parsers and preprocessors.

6. **parse_code**: A convenience function that creates an instance of the CodeParser and parses the provided code using the specified language.

## Usage

To use the `base_parser.py` module as a foundation for language-specific parsers, follow these steps:

1. Import the necessary classes and enums from the module:
   
   ```python
from base_parser import BaseParser, ParseResult, PreprocessorConfig, Language
```

2.  Create a language-specific parser class that inherits from the BaseParser class:

```python
class MyLanguageParser(BaseParser):
    def parse_code(self, code: str, preprocess: bool = True, config: Optional[PreprocessorConfig] = None) -> ParseResult:
        # Implement language-specific parsing logic here
        pass
```

3. Implement the `parse_code` method in the language-specific parser class, which should parse the code and return a `ParseResult` object containing the parsed AST, errors, warnings, tokens, metrics, and language information.
4. Optionally, override or extend other methods from the BaseParser class to customize preprocessing, comment removal, macro expansion, whitespace normalization, directive handling, complexity calculation, or syntax error formatting for the specific language.
5. Use the parse_code convenience function or create an instance of the CodeParser class to parse code using the language-specific parser:
```python
from base_parser import parse_code, Language

code = "// Your code here"
result = parse_code(code, Language.MYLANGUAGE)
```
6. Access the various properties of the ParseResult object to retrieve the analysis results, such as the AST, errors, warnings, tokens, and metrics.

## Extensibility

The base_parser.py module is designed to be extensible, allowing the addition of new language-specific parsers and customization of parsing behavior. To extend the module:

1. Create a new language-specific parser class that inherits from the BaseParser class.
2. Implement the required methods, such as parse_code, and any additional language-specific functionality.
3. Update the CodeParser class to include the new language-specific parser in the parsers dictionary, mapping the language enum to the parser instance.
4. Optionally, customize the preprocessing steps, comment removal, macro expansion, whitespace normalization, directive handling, complexity calculation, or syntax error formatting for the specific language by overriding the corresponding methods in the language-specific parser class.

By following this extensible architecture, the base_parser.py module can serve as a solid foundation for building a multi-language code parsing and analysis system.

## Conclusion

The `base_parser.py` module provides a flexible and extensible framework for code parsing and analysis across different programming languages. By defining common abstractions, utilities, and interfaces, it enables the development of language-specific parsers that can inherit and leverage the core functionality.

With the `base_parser.py` module, you can easily integrate new language parsers, customize parsing behavior, and perform comprehensive code analysis tasks. It forms the foundation of a robust code analysis system that can be extended and adapted to support a wide range of programming languages and analysis requirements.