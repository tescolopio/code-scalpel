"""Analysis Cache - Content-Addressable Cache for Code Analysis Results.

This module provides a caching layer to avoid re-analyzing unchanged code.
Cache keys are SHA-256 hashes of the source code content, ensuring that
identical code always produces cache hits.

Example:
    >>> from code_scalpel.utilities import AnalysisCache
    >>> cache = AnalysisCache()
    >>> 
    >>> # Check cache before expensive operation
    >>> code = "def foo(): return 1"
    >>> cached = cache.get(code, "analysis")
    >>> if cached is None:
    ...     result = expensive_analysis(code)
    ...     cache.set(code, "analysis", result)
    ... else:
    ...     result = cached

The cache supports multiple result types per code hash:
- "analysis": AnalysisResult from analyze_code
- "security": SecurityResult from security_scan  
- "symbolic": SymbolicResult from symbolic_execute
- "tests": TestGenerationResult from generate_unit_tests
"""

import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Import tool version for cache invalidation
try:
    from code_scalpel import __version__ as TOOL_VERSION
except ImportError:
    TOOL_VERSION = "unknown"

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for the analysis cache."""

    # Cache location
    cache_dir: Path | None = None  # None = auto-detect
    use_global_cache: bool = True  # Use ~/.cache/code-scalpel/
    use_local_cache: bool = True  # Use .scalpel_cache/

    # Cache behavior
    max_entries: int = 10000  # Maximum cache entries
    max_size_mb: int = 500  # Maximum cache size in MB
    ttl_seconds: int = 86400 * 7  # Time-to-live: 7 days

    # Serialization
    use_pickle: bool = True  # True = pickle (fast), False = JSON (portable)

    # Performance
    enabled: bool = True  # Master switch to disable caching


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    result: Any
    timestamp: float
    code_hash: str
    result_type: str
    config_hash: str = ""
    hits: int = 0


@dataclass
class CacheStats:
    """Statistics about cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    size_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total_entries": self.total_entries,
            "size_bytes": self.size_bytes,
            "hit_rate": f"{self.hit_rate:.2%}",
        }


class AnalysisCache:
    """Content-addressable cache for code analysis results.

    This cache stores analysis results keyed by SHA-256 hash of the source
    code. This ensures:

    1. Identical code always hits the cache
    2. Any code change invalidates the cache
    3. No need for file watching or timestamps

    The cache is stored on disk to persist across sessions.

    Cache Key Structure:
        {code_hash}_{result_type}_{config_hash}.cache

    Example:
        a1b2c3d4..._{analysis}_{default}.cache
    """

    VERSION = "1.0"  # Cache format version (invalidate on breaking changes)

    def __init__(self, config: CacheConfig | None = None):
        """Initialize the cache.

        Args:
            config: Cache configuration. Uses defaults if None.
        """
        self.config = config or CacheConfig()
        self.stats = CacheStats()
        self._memory_cache: dict[str, CacheEntry] = {}  # In-memory LRU
        self._cache_dir: Path | None = None

        if self.config.enabled:
            self._cache_dir = self._resolve_cache_dir()
            if self._cache_dir:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Cache initialized at {self._cache_dir}")

    def _resolve_cache_dir(self) -> Path | None:
        """Resolve the cache directory location."""
        if self.config.cache_dir:
            return self.config.cache_dir

        # Try local cache first (project-specific)
        if self.config.use_local_cache:
            local_cache = Path.cwd() / ".scalpel_cache"
            if local_cache.exists() or self._can_create_dir(local_cache):
                return local_cache

        # Fall back to global cache
        if self.config.use_global_cache:
            if os.name == "nt":  # Windows
                base = Path(
                    os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
                )
            else:  # Unix
                base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            return base / "code-scalpel"

        return None

    def _can_create_dir(self, path: Path) -> bool:
        """Check if we can create a directory at the given path."""
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except (PermissionError, OSError):
            return False

    def _hash_code(self, code: str) -> str:
        """Generate SHA-256 hash of code content."""
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def _hash_config(self, config: dict[str, Any] | None = None) -> str:
        """Generate hash of configuration that affects analysis.

        The hash includes the tool version to automatically invalidate
        cache entries when code-scalpel is upgraded.

        Cache Key Formula: SHA256(config + tool_version)
        """
        # Always include tool version for automatic invalidation on upgrades
        effective_config = {"_tool_version": TOOL_VERSION}
        if config:
            effective_config.update(config)

        # Sort keys for consistent hashing
        config_str = json.dumps(effective_config, sort_keys=True)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]

    def _cache_key(self, code_hash: str, result_type: str, config_hash: str) -> str:
        """Generate cache key from components."""
        return f"{code_hash}_{result_type}_{config_hash}"

    def _cache_path(self, cache_key: str) -> Path | None:
        """Get the file path for a cache entry."""
        if self._cache_dir is None:
            return None
        ext = ".pkl" if self.config.use_pickle else ".json"
        return self._cache_dir / f"v{self.VERSION}" / f"{cache_key}{ext}"

    def get(
        self,
        code: str,
        result_type: str,
        config: dict[str, Any] | None = None,
    ) -> Any | None:
        """Retrieve a cached result.

        Args:
            code: Source code that was analyzed
            result_type: Type of result ("analysis", "security", "symbolic", "tests")
            config: Configuration used for analysis (affects cache key)

        Returns:
            Cached result if found and valid, None otherwise
        """
        if not self.config.enabled:
            return None

        code_hash = self._hash_code(code)
        config_hash = self._hash_config(config)
        cache_key = self._cache_key(code_hash, result_type, config_hash)

        # Check in-memory cache first
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]
            if self._is_valid(entry):
                entry.hits += 1
                self.stats.hits += 1
                logger.debug(
                    f"Cache hit (memory): {result_type} for {code_hash[:8]}..."
                )
                return entry.result

        # Check disk cache
        cache_path = self._cache_path(cache_key)
        if cache_path and cache_path.exists():
            try:
                entry = self._load_entry(cache_path)
                if entry and self._is_valid(entry):
                    # Promote to memory cache
                    self._memory_cache[cache_key] = entry
                    entry.hits += 1
                    self.stats.hits += 1
                    logger.debug(
                        f"Cache hit (disk): {result_type} for {code_hash[:8]}..."
                    )
                    return entry.result
            except Exception as e:
                logger.warning(f"Failed to load cache entry: {e}")

        self.stats.misses += 1
        logger.debug(f"Cache miss: {result_type} for {code_hash[:8]}...")
        return None

    def set(
        self,
        code: str,
        result_type: str,
        result: Any,
        config: dict[str, Any] | None = None,
    ) -> bool:
        """Store a result in the cache.

        Args:
            code: Source code that was analyzed
            result_type: Type of result
            result: The analysis result to cache
            config: Configuration used for analysis

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.config.enabled:
            return False

        code_hash = self._hash_code(code)
        config_hash = self._hash_config(config)
        cache_key = self._cache_key(code_hash, result_type, config_hash)

        entry = CacheEntry(
            result=result,
            timestamp=time.time(),
            code_hash=code_hash,
            result_type=result_type,
            config_hash=config_hash,
        )

        # Store in memory
        self._memory_cache[cache_key] = entry

        # Store on disk
        cache_path = self._cache_path(cache_key)
        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_entry(cache_path, entry)
                self.stats.total_entries += 1
                logger.debug(f"Cached: {result_type} for {code_hash[:8]}...")
                return True
            except Exception as e:
                logger.warning(f"Failed to save cache entry: {e}")

        return False

    def _is_valid(self, entry: CacheEntry) -> bool:
        """Check if a cache entry is still valid."""
        if self.config.ttl_seconds <= 0:
            return True
        age = time.time() - entry.timestamp
        return age < self.config.ttl_seconds

    def _load_entry(self, path: Path) -> CacheEntry | None:
        """Load a cache entry from disk."""
        try:
            if self.config.use_pickle:
                with open(path, "rb") as f:
                    return pickle.load(f)
            else:
                with open(path, "r") as f:
                    data = json.load(f)
                    return CacheEntry(**data)
        except Exception:
            return None

    def _save_entry(self, path: Path, entry: CacheEntry) -> None:
        """Save a cache entry to disk."""
        if self.config.use_pickle:
            with open(path, "wb") as f:
                pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, "w") as f:
                # Convert to JSON-serializable dict
                data = {
                    "result": self._to_json_serializable(entry.result),
                    "timestamp": entry.timestamp,
                    "code_hash": entry.code_hash,
                    "result_type": entry.result_type,
                    "config_hash": entry.config_hash,
                    "hits": entry.hits,
                }
                json.dump(data, f)

    def _to_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable form."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return obj

    def invalidate(self, code: str, result_type: str | None = None) -> int:
        """Invalidate cache entries for specific code.

        Args:
            code: Source code to invalidate
            result_type: Specific result type, or None for all types

        Returns:
            Number of entries invalidated
        """
        code_hash = self._hash_code(code)
        count = 0

        # Invalidate memory cache
        keys_to_remove = [
            k
            for k in self._memory_cache
            if k.startswith(code_hash) and (result_type is None or result_type in k)
        ]
        for key in keys_to_remove:
            del self._memory_cache[key]
            count += 1

        # Invalidate disk cache
        if self._cache_dir:
            version_dir = self._cache_dir / f"v{self.VERSION}"
            if version_dir.exists():
                pattern = f"{code_hash}_{result_type or '*'}_*"
                for path in version_dir.glob(pattern + ".*"):
                    path.unlink()
                    count += 1

        self.stats.evictions += count
        return count

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        count = len(self._memory_cache)
        self._memory_cache.clear()

        if self._cache_dir and self._cache_dir.exists():
            import shutil

            for item in self._cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                    count += 1
                elif item.is_file():
                    item.unlink()
                    count += 1

        self.stats = CacheStats()
        logger.info(f"Cache cleared: {count} entries removed")
        return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        # Update entry count
        self.stats.total_entries = len(self._memory_cache)

        # Calculate disk size
        if self._cache_dir and self._cache_dir.exists():
            total_size = sum(
                f.stat().st_size for f in self._cache_dir.rglob("*") if f.is_file()
            )
            self.stats.size_bytes = total_size

        return self.stats


# Global cache instance (singleton pattern)
_global_cache: AnalysisCache | None = None


def get_cache(config: CacheConfig | None = None) -> AnalysisCache:
    """Get the global cache instance.

    Args:
        config: Cache configuration (only used on first call)

    Returns:
        The global AnalysisCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = AnalysisCache(config)
    return _global_cache


def reset_cache() -> None:
    """Reset the global cache instance."""
    global _global_cache
    if _global_cache:
        _global_cache.clear()
    _global_cache = None
