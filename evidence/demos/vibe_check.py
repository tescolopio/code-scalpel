"""
Vibe Check Demo: Hidden SQL Injection

This demo proves that Code Scalpel tracks TAINT FLOW, not just keywords.
A regex-based linter would miss this because the dangerous input is hidden
through multiple variable assignments.

Run:
    code-scalpel scan demos/vibe_check.py

Expected: Code Scalpel detects SQL Injection at cursor.execute() 
          by tracing: request.args -> user_id -> query_base -> final_query
"""
import sqlite3
from flask import request


def get_user_data():
    """Fetch user data - contains hidden SQL injection vulnerability."""
    # SOURCE: User input (Tainted)
    user_id = request.args.get("id")

    # PROPAGATION: Hiding the taint through variable indirection
    # A regex linter looking for "request.args" near "execute" would miss this
    part1 = "SELECT * FROM users WHERE id = '"
    part2 = "'"
    query_base = part1 + user_id  # Taint flows here

    # More indirection
    final_query = query_base + part2  # Taint flows here too

    # SINK: SQL Execution with tainted data
    conn = sqlite3.connect("db.sqlite")
    cursor = conn.cursor()

    # Code Scalpel should detect this despite 3 levels of indirection
    # Taint path: request.args.get('id') -> user_id -> query_base -> final_query
    cursor.execute(final_query)  # VULNERABILITY: CWE-89 SQL Injection

    return cursor.fetchall()


def safe_version():
    """The correct way - parameterized queries."""
    user_id = request.args.get("id")

    conn = sqlite3.connect("db.sqlite")
    cursor = conn.cursor()

    # Parameterized query - no taint flow to execute()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))

    return cursor.fetchall()
