#!/bin/bash

# Define the base directory
base_dir="src/code_parser"

# Define the languages and their corresponding analyzers/linters
declare -A analyzers=(
  ["java_parsers"]="javalang SpotBugs PMD Checkstyle FindSecBugs ErrorProne Infer SonarQube JArchitect"
  ["python_parsers"]="pylint flake8 mypy bandit prospector pycodestyle pydocstyle" 
  ["javascript_parsers"]="eslint jshint prettier standard"
  ["csharp_parsers"]="Roslyn-Analyzers StyleCop ReSharper SonarQube"
  ["go_parsers"]="gofmt golint govet staticcheck"
  ["cpp_parsers"]="Clang-Static-Analyzer Cppcheck SonarQube"
  ["php_parsers"]="PHPCS PHPStan Psalm"
  ["swift_parsers"]="SwiftLint Tailor"
  ["kotlin_parsers"]="ktlint Detekt"
  ["ruby_parsers"]="RuboCop Reek"
)

# Create the base directory if it doesn't exist
mkdir -p "$base_dir"

# Loop through the languages
for language in "${!analyzers[@]}"; do
  # Create the language directory
  mkdir -p "$base_dir/$language"

  # Get the analyzers for the current language
  lang_analyzers="${analyzers[$language]}"

  # Loop through the analyzers and create the files
  for analyzer in $lang_analyzers; do
    # Determine the appropriate file extension based on language
    case "$language" in
      "csharp_parsers") extension=".py" ;;
      "go_parsers") extension=".py" ;;
      "cpp_parsers") extension=".py" ;;  # Or .cc, .cxx, etc.
      "php_parsers") extension=".py" ;;
      "swift_parsers") extension=".py" ;;
      "kotlin_parsers") extension=".py" ;;
      "ruby_parsers") extension=".py" ;;
      *) extension=".py" ;;  # Default to .py
    esac

    # Use 'touch' for most files, but 'echo' for Python to add a shebang
    if [[ "$extension" == ".py" ]]; then
      echo "#!/usr/bin/env python3" > "$base_dir/$language/${language}_${analyzer}${extension}"
    else
      touch "$base_dir/$language/${language}_${analyzer}${extension}"
    fi
  done
done