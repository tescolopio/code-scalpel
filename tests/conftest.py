"""
Pytest configuration and fixtures for Code Scalpel tests.
"""
import os
import sys

# Add the src directory to the path so tests can import code_scalpel
# This allows tests to run both before and after pip install
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
