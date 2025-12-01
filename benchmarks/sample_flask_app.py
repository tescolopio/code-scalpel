"""
Sample Flask application for benchmarking CodeAnalyzer.

This simulates a real-world Flask web application with:
- Multiple routes
- Database models
- Helper functions
- Some intentional dead code
"""

from flask import Flask, request, jsonify, render_template
from functools import wraps
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Setup logging
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
DEBUG = True
SECRET_KEY = 'development-secret-key'
DATABASE_URL = 'sqlite:///app.db'

# ----- Models -----

class User:
    """User model representing application users."""
    
    def __init__(self, id: int, username: str, email: str):
        self.id = id
        self.username = username
        self.email = email
        self.created_at = datetime.now()
        self.is_active = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active
        }
    
    def update_email(self, new_email: str) -> None:
        self.email = new_email
    
    def deactivate(self) -> None:
        self.is_active = False
    
    # Dead code: unused method
    def deprecated_method(self) -> str:
        """This method is no longer used."""
        return f"Deprecated for user {self.username}"


class Post:
    """Blog post model."""
    
    def __init__(self, id: int, title: str, content: str, author_id: int):
        self.id = id
        self.title = title
        self.content = content
        self.author_id = author_id
        self.created_at = datetime.now()
        self.views = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'author_id': self.author_id,
            'created_at': self.created_at.isoformat(),
            'views': self.views
        }
    
    def increment_views(self) -> None:
        self.views += 1


# ----- Database simulation -----

users_db: Dict[int, User] = {}
posts_db: Dict[int, Post] = {}

def init_db():
    """Initialize database with sample data."""
    global users_db, posts_db
    
    users_db[1] = User(1, 'admin', 'admin@example.com')
    users_db[2] = User(2, 'user1', 'user1@example.com')
    
    posts_db[1] = Post(1, 'Welcome', 'Welcome to our blog!', 1)
    posts_db[2] = Post(2, 'Tutorial', 'Getting started guide', 1)


def get_user(user_id: int) -> Optional[User]:
    """Get user by ID."""
    return users_db.get(user_id)


def get_post(post_id: int) -> Optional[Post]:
    """Get post by ID."""
    return posts_db.get(post_id)


def create_user(username: str, email: str) -> User:
    """Create a new user."""
    new_id = max(users_db.keys(), default=0) + 1
    user = User(new_id, username, email)
    users_db[new_id] = user
    return user


def create_post(title: str, content: str, author_id: int) -> Post:
    """Create a new post."""
    new_id = max(posts_db.keys(), default=0) + 1
    post = Post(new_id, title, content, author_id)
    posts_db[new_id] = post
    return post


# Dead code: unused function
def unused_database_cleanup():
    """This function is never called."""
    old_posts = []
    for post_id, post in posts_db.items():
        days_old = (datetime.now() - post.created_at).days
        if days_old > 365:
            old_posts.append(post_id)
    for post_id in old_posts:
        del posts_db[post_id]


# ----- Decorators -----

def require_auth(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authentication required'}), 401
        # Simple token validation
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Invalid token format'}), 401
        return f(*args, **kwargs)
    return decorated


def rate_limit(max_requests: int = 100):
    """Rate limiting decorator."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # Rate limiting logic would go here
            return f(*args, **kwargs)
        return decorated
    return decorator


# Dead code: unused decorator
def deprecated_decorator(f):
    """This decorator is no longer used."""
    @wraps(f)
    def decorated(*args, **kwargs):
        logger.warning(f"Deprecated endpoint: {f.__name__}")
        return f(*args, **kwargs)
    return decorated


# ----- Routes -----

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/api/health')
def health_check():
    """API health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


@app.route('/api/users', methods=['GET'])
@rate_limit(max_requests=50)
def list_users():
    """List all users."""
    users = [user.to_dict() for user in users_db.values()]
    return jsonify({'users': users})


@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user_route(user_id: int):
    """Get a specific user."""
    user = get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(user.to_dict())


@app.route('/api/users', methods=['POST'])
@require_auth
def create_user_route():
    """Create a new user."""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    username = data.get('username')
    email = data.get('email')
    
    if not username or not email:
        return jsonify({'error': 'Username and email required'}), 400
    
    user = create_user(username, email)
    return jsonify(user.to_dict()), 201


@app.route('/api/posts', methods=['GET'])
def list_posts():
    """List all posts."""
    posts = [post.to_dict() for post in posts_db.values()]
    return jsonify({'posts': posts})


@app.route('/api/posts/<int:post_id>', methods=['GET'])
def get_post_route(post_id: int):
    """Get a specific post."""
    post = get_post(post_id)
    if not post:
        return jsonify({'error': 'Post not found'}), 404
    post.increment_views()
    return jsonify(post.to_dict())


@app.route('/api/posts', methods=['POST'])
@require_auth
def create_post_route():
    """Create a new post."""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    title = data.get('title')
    content = data.get('content')
    author_id = data.get('author_id')
    
    if not title or not content or not author_id:
        return jsonify({'error': 'Title, content, and author_id required'}), 400
    
    # Verify author exists
    author = get_user(author_id)
    if not author:
        return jsonify({'error': 'Author not found'}), 400
    
    post = create_post(title, content, author_id)
    return jsonify(post.to_dict()), 201


# ----- Helper Functions -----

def validate_email(email: str) -> bool:
    """Validate email format."""
    if not email:
        return False
    if '@' not in email:
        return False
    if '.' not in email.split('@')[1]:
        return False
    return True


def format_response(data: Any, status: str = 'success') -> Dict[str, Any]:
    """Format API response."""
    return {
        'status': status,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }


# Dead code: unused helper
def unused_helper_function(items: List[Any]) -> List[Any]:
    """This helper function is never called."""
    result = []
    for item in items:
        if item is not None:
            result.append(item)
    return result


def calculate_statistics(posts: List[Post]) -> Dict[str, int]:
    """Calculate post statistics."""
    total_views = sum(post.views for post in posts)
    total_posts = len(posts)
    avg_views = total_views // total_posts if total_posts > 0 else 0
    
    return {
        'total_posts': total_posts,
        'total_views': total_views,
        'average_views': avg_views
    }


# ----- Error Handlers -----

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    logger.error(f"Server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# ----- Application Entry Point -----

if __name__ == '__main__':
    init_db()
    app.run(debug=DEBUG, host='0.0.0.0', port=5000)
