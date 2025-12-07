"""
Django Security Demo: ORM vs Raw SQL Vulnerabilities

This demo shows how Code Scalpel differentiates between safe ORM usage
and dangerous raw SQL in Django applications.

Target: Django developers
Proves: Code Scalpel understands framework patterns

Run:
    code-scalpel scan demos/real_world/django_views.py
"""
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.db import connection
from django.db.models import Q
from django.views import View
from django.utils.html import escape
import os


# Assume these models exist
class User:
    objects = None

class Product:
    objects = None


# =============================================================================
# VULNERABLE: Raw SQL with string formatting
# =============================================================================
def search_users_vulnerable(request):
    """
    Search users using raw SQL - VULNERABLE.
    
    Code Scalpel should detect: request.GET -> search_term -> query -> cursor.execute()
    """
    search_term = request.GET.get('q', '')
    
    # BAD: Raw SQL with string formatting
    with connection.cursor() as cursor:
        query = f"SELECT * FROM users WHERE name LIKE '%{search_term}%'"
        cursor.execute(query)  # CWE-89: SQL Injection
        results = cursor.fetchall()
    
    return JsonResponse({'users': results})


def get_user_by_id_vulnerable(request, user_id):
    """
    Get user by ID using raw SQL - VULNERABLE.
    
    Even though user_id looks like an int, it could be manipulated.
    """
    # BAD: Trusting URL parameter in raw SQL
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")  # CWE-89
        user = cursor.fetchone()
    
    return JsonResponse({'user': user})


# =============================================================================
# SAFE: Django ORM (parameterized by default)
# =============================================================================
def search_users_safe(request):
    """
    Search users using Django ORM - SAFE.
    
    Django's ORM automatically parameterizes queries.
    Code Scalpel should NOT flag this.
    """
    search_term = request.GET.get('q', '')
    
    # GOOD: ORM handles parameterization
    users = User.objects.filter(
        Q(name__icontains=search_term) | Q(email__icontains=search_term)
    )
    
    return JsonResponse({'users': list(users.values())})


def get_user_safe(request, user_id):
    """
    Get user by ID using ORM - SAFE.
    """
    # GOOD: get_object_or_404 uses parameterized ORM query
    user = get_object_or_404(User, pk=user_id)
    return JsonResponse({'user': {'id': user.id, 'name': user.name}})


# =============================================================================
# SAFE: Raw SQL with parameters
# =============================================================================
def search_users_raw_safe(request):
    """
    Raw SQL but with proper parameterization - SAFE.
    
    Code Scalpel should recognize the parameterized pattern.
    """
    search_term = request.GET.get('q', '')
    
    # GOOD: Parameterized raw SQL
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM users WHERE name LIKE %s",
            [f'%{search_term}%']  # Parameters passed separately
        )
        results = cursor.fetchall()
    
    return JsonResponse({'users': results})


# =============================================================================
# VULNERABLE: XSS in template context
# =============================================================================
def user_profile_vulnerable(request, username):
    """
    Render user profile with unsanitized data - VULNERABLE.
    
    The |safe filter bypasses Django's auto-escaping.
    """
    user = get_object_or_404(User, username=username)
    
    # BAD: Passing unsanitized HTML to template marked as safe
    # Template uses: {{ bio|safe }} which is XSS
    return render(request, 'profile.html', {
        'user': user,
        'bio_html': user.bio,  # If template uses |safe, this is XSS
    })


def user_profile_safe(request, username):
    """
    Render user profile with proper escaping - SAFE.
    """
    user = get_object_or_404(User, username=username)
    
    # GOOD: Explicitly escape user content
    return render(request, 'profile.html', {
        'user': user,
        'bio_html': escape(user.bio),  # Escaped before template
    })


# =============================================================================
# VULNERABLE: File operations
# =============================================================================
def download_file_vulnerable(request):
    """
    Download file with user-controlled path - VULNERABLE.
    """
    filename = request.GET.get('file', '')
    
    # BAD: Path traversal
    filepath = os.path.join('/var/www/uploads', filename)
    
    with open(filepath, 'rb') as f:  # CWE-22: Path Traversal
        return HttpResponse(f.read(), content_type='application/octet-stream')


def download_file_safe(request):
    """
    Download file with path validation - SAFE.
    """
    filename = request.GET.get('file', '')
    base_path = '/var/www/uploads'
    
    # GOOD: Resolve and validate path
    filepath = os.path.realpath(os.path.join(base_path, filename))
    
    if not filepath.startswith(base_path):
        return HttpResponse("Access denied", status=403)
    
    with open(filepath, 'rb') as f:
        return HttpResponse(f.read(), content_type='application/octet-stream')


# =============================================================================
# Expected Code Scalpel Output:
#
# Found 4 vulnerability(ies):
#   1. SQL Injection (CWE-89) at line 32
#   2. SQL Injection (CWE-89) at line 46
#   3. XSS (CWE-79) at line 98 (requires template analysis)
#   4. Path Traversal (CWE-22) at line 117
#
# Safe patterns identified:
#   - ORM queries (auto-parameterized)
#   - Parameterized raw SQL
#   - escape() usage
#   - Path validation
# =============================================================================
