"""
FastAPI Security Demo: Real-World API Vulnerabilities

This demo shows Code Scalpel detecting vulnerabilities in a realistic
FastAPI application with multiple endpoints and database operations.

Target: Backend developers building REST APIs
Proves: Code Scalpel works on real-world web frameworks, not just snippets

Run:
    code-scalpel scan demos/real_world/fastapi_app.py
"""
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import HTMLResponse
import sqlite3
import subprocess
import os

app = FastAPI(title="Vulnerable Demo API")


# Database setup
def get_db():
    return sqlite3.connect("app.db")


# =============================================================================
# VULNERABILITY 1: SQL Injection in Search
# =============================================================================
@app.get("/api/users/search")
async def search_users(q: str = Query(..., description="Search query")):
    """
    Search users by name.

    VULNERABILITY: Direct string concatenation in SQL query.
    Code Scalpel should detect: USER_INPUT -> q -> query -> cursor.execute()
    """
    db = get_db()
    cursor = db.cursor()

    # BAD: SQL Injection
    query = f"SELECT * FROM users WHERE name LIKE '%{q}%'"
    cursor.execute(query)  # CWE-89: SQL Injection

    return {"users": cursor.fetchall()}


# =============================================================================
# VULNERABILITY 2: Command Injection in Export
# =============================================================================
@app.post("/api/export")
async def export_data(filename: str):
    """
    Export database to a file.

    VULNERABILITY: User-controlled filename in shell command.
    Code Scalpel should detect: filename -> cmd -> subprocess.run()
    """
    # BAD: Command Injection
    cmd = f"sqlite3 app.db .dump > /tmp/{filename}"
    subprocess.run(cmd, shell=True)  # CWE-78: Command Injection

    return {"status": "exported", "file": filename}


# =============================================================================
# VULNERABILITY 3: XSS in Profile Display
# =============================================================================
@app.get("/profile/{user_id}", response_class=HTMLResponse)
async def get_profile(user_id: int, request: Request):
    """
    Display user profile page.

    VULNERABILITY: User data rendered directly into HTML.
    Code Scalpel should detect taint flow to HTML response.
    """
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT name, bio FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()

    if not user:
        raise HTTPException(status_code=404)

    name, bio = user

    # BAD: XSS - user bio rendered without escaping
    html = f"""
    <html>
        <body>
            <h1>{name}'s Profile</h1>
            <div class="bio">{bio}</div>
        </body>
    </html>
    """  # CWE-79: XSS

    return HTMLResponse(content=html)


# =============================================================================
# VULNERABILITY 4: Path Traversal in File Download
# =============================================================================
@app.get("/api/files/{filename}")
async def download_file(filename: str):
    """
    Download a file from the uploads directory.

    VULNERABILITY: User-controlled path without validation.
    Code Scalpel should detect: filename -> filepath -> open()
    """
    # BAD: Path Traversal
    filepath = f"/app/uploads/{filename}"  # User could pass "../../../etc/passwd"

    if os.path.exists(filepath):
        with open(filepath, "r") as f:  # CWE-22: Path Traversal
            return {"content": f.read()}

    raise HTTPException(status_code=404)


# =============================================================================
# SAFE ENDPOINTS (for comparison)
# =============================================================================
@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    """
    Get user by ID - SAFE because parameterized query.
    Code Scalpel should NOT flag this.
    """
    db = get_db()
    cursor = db.cursor()

    # GOOD: Parameterized query
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))

    return {"user": cursor.fetchone()}


@app.get("/api/health")
async def health_check():
    """Simple health check - no user input, no vulnerabilities."""
    return {"status": "healthy", "version": "1.0.0"}


# =============================================================================
# Expected Code Scalpel Output:
#
# Found 4 vulnerability(ies):
#   1. SQL Injection (CWE-89) at line 35
#      Taint path: q -> query -> cursor.execute()
#   2. Command Injection (CWE-78) at line 49
#      Taint path: filename -> cmd -> subprocess.run()
#   3. XSS (CWE-79) at line 73
#      Taint path: bio -> html -> HTMLResponse
#   4. Path Traversal (CWE-22) at line 89
#      Taint path: filename -> filepath -> open()
# =============================================================================
