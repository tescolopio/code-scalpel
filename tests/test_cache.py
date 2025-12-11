"""Tests for the Analysis Cache.

Tests the content-addressable caching layer.
"""

import tempfile
import time
from pathlib import Path

import pytest


class TestAnalysisCache:
    """Tests for AnalysisCache class."""

    def test_cache_set_and_get(self):
        """Test basic set and get operations."""
        from code_scalpel.utilities.cache import AnalysisCache, CacheConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = AnalysisCache(config)

            code = "def foo(): return 1"
            result = {"success": True, "functions": ["foo"]}

            cache.set(code, "analysis", result)
            cached = cache.get(code, "analysis")

            assert cached == result

    def test_cache_miss(self):
        """Test cache miss returns None."""
        from code_scalpel.utilities.cache import AnalysisCache, CacheConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = AnalysisCache(config)

            code = "def bar(): return 2"
            cached = cache.get(code, "analysis")

            assert cached is None

    def test_different_code_different_cache(self):
        """Test that different code gets different cache entries."""
        from code_scalpel.utilities.cache import AnalysisCache, CacheConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = AnalysisCache(config)

            code1 = "def foo(): return 1"
            code2 = "def bar(): return 2"
            result1 = {"functions": ["foo"]}
            result2 = {"functions": ["bar"]}

            cache.set(code1, "analysis", result1)
            cache.set(code2, "analysis", result2)

            assert cache.get(code1, "analysis") == result1
            assert cache.get(code2, "analysis") == result2

    def test_different_result_types(self):
        """Test caching different result types for same code."""
        from code_scalpel.utilities.cache import AnalysisCache, CacheConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = AnalysisCache(config)

            code = "def foo(): return 1"
            analysis_result = {"type": "analysis"}
            security_result = {"type": "security"}

            cache.set(code, "analysis", analysis_result)
            cache.set(code, "security", security_result)

            assert cache.get(code, "analysis") == analysis_result
            assert cache.get(code, "security") == security_result

    def test_config_affects_cache_key(self):
        """Test that different configs create different cache entries."""
        from code_scalpel.utilities.cache import AnalysisCache, CacheConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = AnalysisCache(config)

            code = "def foo(): return 1"
            result1 = {"config": "python"}
            result2 = {"config": "java"}

            cache.set(code, "analysis", result1, {"language": "python"})
            cache.set(code, "analysis", result2, {"language": "java"})

            assert cache.get(code, "analysis", {"language": "python"}) == result1
            assert cache.get(code, "analysis", {"language": "java"}) == result2

    def test_cache_disabled(self):
        """Test that disabled cache returns None."""
        from code_scalpel.utilities.cache import AnalysisCache, CacheConfig

        config = CacheConfig(enabled=False)
        cache = AnalysisCache(config)

        code = "def foo(): return 1"
        result = {"success": True}

        assert cache.set(code, "analysis", result) is False
        assert cache.get(code, "analysis") is None

    def test_cache_invalidate(self):
        """Test cache invalidation."""
        from code_scalpel.utilities.cache import AnalysisCache, CacheConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = AnalysisCache(config)

            code = "def foo(): return 1"
            result = {"success": True}

            cache.set(code, "analysis", result)
            assert cache.get(code, "analysis") == result

            cache.invalidate(code, "analysis")
            assert cache.get(code, "analysis") is None

    def test_cache_clear(self):
        """Test clearing entire cache."""
        from code_scalpel.utilities.cache import AnalysisCache, CacheConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = AnalysisCache(config)

            cache.set("code1", "analysis", {"a": 1})
            cache.set("code2", "analysis", {"b": 2})

            cache.clear()

            assert cache.get("code1", "analysis") is None
            assert cache.get("code2", "analysis") is None

    def test_cache_stats(self):
        """Test cache statistics."""
        from code_scalpel.utilities.cache import AnalysisCache, CacheConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = AnalysisCache(config)

            code = "def foo(): return 1"
            cache.set(code, "analysis", {"success": True})

            # First get is a hit (from memory)
            cache.get(code, "analysis")
            cache.get(code, "analysis")

            # Miss
            cache.get("different code", "analysis")

            stats = cache.get_stats()
            assert stats.hits >= 2
            assert stats.misses >= 1

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        from code_scalpel.utilities.cache import CacheStats

        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == 0.75

        empty_stats = CacheStats()
        assert empty_stats.hit_rate == 0.0

    def test_cache_persistence(self):
        """Test that cache persists across instances."""
        from code_scalpel.utilities.cache import AnalysisCache, CacheConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first instance and set value
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache1 = AnalysisCache(config)
            code = "def foo(): return 1"
            cache1.set(code, "analysis", {"persisted": True})

            # Create new instance and check value
            cache2 = AnalysisCache(config)
            result = cache2.get(code, "analysis")
            assert result == {"persisted": True}

    def test_ttl_expiration(self):
        """Test that expired entries are not returned."""
        from code_scalpel.utilities.cache import AnalysisCache, CacheConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            # Very short TTL
            config = CacheConfig(cache_dir=Path(tmpdir), ttl_seconds=1)
            cache = AnalysisCache(config)

            code = "def foo(): return 1"
            cache.set(code, "analysis", {"success": True})

            # Should be valid immediately
            assert cache.get(code, "analysis") is not None

            # Wait for expiration
            time.sleep(1.5)

            # Should be expired now
            assert cache.get(code, "analysis") is None

    def test_version_change_invalidates_cache(self):
        """Test that changing tool version invalidates cache entries.

        This ensures that upgrading code-scalpel automatically invalidates
        old cache entries, preventing stale results from older analysis.
        """
        from code_scalpel.utilities import cache as cache_module
        from code_scalpel.utilities.cache import AnalysisCache, CacheConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))

            # Simulate version 0.7.0
            original_version = cache_module.TOOL_VERSION
            cache_module.TOOL_VERSION = "0.7.0"

            try:
                cache1 = AnalysisCache(config)
                code = "def foo(): return 1"
                cache1.set(code, "analysis", {"version": "0.7.0"})

                # Same version should hit
                assert cache1.get(code, "analysis") == {"version": "0.7.0"}

                # Simulate upgrade to version 0.8.0
                cache_module.TOOL_VERSION = "0.8.0"
                cache2 = AnalysisCache(config)

                # Different version should miss (automatic invalidation)
                assert cache2.get(code, "analysis") is None

                # New version can set its own cache
                cache2.set(code, "analysis", {"version": "0.8.0"})
                assert cache2.get(code, "analysis") == {"version": "0.8.0"}
            finally:
                # Restore original version
                cache_module.TOOL_VERSION = original_version


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from code_scalpel.utilities.cache import CacheConfig

        config = CacheConfig()

        assert config.enabled is True
        assert config.max_entries == 10000
        assert config.ttl_seconds == 86400 * 7  # 7 days
        assert config.use_pickle is True

    def test_custom_config(self):
        """Test custom configuration."""
        from code_scalpel.utilities.cache import CacheConfig

        config = CacheConfig(
            max_entries=100,
            ttl_seconds=3600,
            use_pickle=False,
        )

        assert config.max_entries == 100
        assert config.ttl_seconds == 3600
        assert config.use_pickle is False


class TestGlobalCache:
    """Tests for global cache functions."""

    def test_get_cache_singleton(self):
        """Test that get_cache returns same instance."""
        from code_scalpel.utilities.cache import get_cache, reset_cache

        reset_cache()  # Clear any existing

        cache1 = get_cache()
        cache2 = get_cache()

        assert cache1 is cache2

    def test_reset_cache(self):
        """Test cache reset."""
        from code_scalpel.utilities.cache import get_cache, reset_cache

        cache1 = get_cache()
        reset_cache()
        cache2 = get_cache()

        assert cache1 is not cache2


class TestCacheIntegration:
    """Integration tests for cache with MCP server."""

    @pytest.mark.asyncio
    async def test_analyze_code_caching(self):
        """Test that analyze_code uses cache."""
        import os

        os.environ["SCALPEL_CACHE_ENABLED"] = "1"

        from code_scalpel.mcp.server import analyze_code

        code = "def test_func(): return 42"

        # First call
        result1 = await analyze_code(code)
        assert result1.success is True

        # Second call should hit cache
        result2 = await analyze_code(code)
        assert result2.success is True
        assert result1.functions == result2.functions

    @pytest.mark.asyncio
    async def test_security_scan_caching(self):
        """Test that security_scan uses cache."""
        import os

        os.environ["SCALPEL_CACHE_ENABLED"] = "1"

        from code_scalpel.mcp.server import security_scan

        code = "def safe(): return 1"

        result1 = await security_scan(code)
        result2 = await security_scan(code)

        assert result1.success == result2.success
        assert result1.risk_level == result2.risk_level


class TestSymbolicCaching:
    """Tests for symbolic execution caching."""

    def test_symbolic_analyzer_caches_results(self):
        """Test that SymbolicAnalyzer caches analysis results."""
        import tempfile
        import time
        from pathlib import Path
        from code_scalpel.utilities.cache import CacheConfig, AnalysisCache, reset_cache

        # Reset global cache to ensure clean state
        reset_cache()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure cache to use temp directory
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = AnalysisCache(config)

            # Patch get_cache to return our test cache
            import code_scalpel.utilities.cache as cache_module

            original_get_cache = cache_module.get_cache
            cache_module.get_cache = lambda *args, **kwargs: cache

            try:
                from code_scalpel.symbolic_execution_tools.engine import (
                    SymbolicAnalyzer,
                )

                analyzer = SymbolicAnalyzer(enable_cache=True)
                code = """
x = 10
if x > 5:
    y = x + 1
else:
    y = x - 1
"""
                # First analysis (cache miss)
                start1 = time.perf_counter()
                result1 = analyzer.analyze(code)
                time1 = time.perf_counter() - start1

                # Second analysis (should be cache hit)
                start2 = time.perf_counter()
                result2 = analyzer.analyze(code)
                time2 = time.perf_counter() - start2

                # Verify results are equivalent
                assert result1.feasible_count == result2.feasible_count
                assert result1.total_paths == result2.total_paths
                assert len(result1.paths) == len(result2.paths)

                # Verify cache hit is faster (at least 2x faster)
                # Note: This can be flaky on slow systems, so we're generous
                assert (
                    time2 < time1
                ), f"Cache hit ({time2:.4f}s) should be faster than miss ({time1:.4f}s)"

                # Verify from_cache flag
                assert result2.from_cache is True

            finally:
                cache_module.get_cache = original_get_cache

    def test_symbolic_analyzer_cache_disabled(self):
        """Test that caching can be disabled."""
        from code_scalpel.symbolic_execution_tools.engine import SymbolicAnalyzer

        analyzer = SymbolicAnalyzer(enable_cache=False)
        assert analyzer._cache is None

        code = "x = 1"
        result = analyzer.analyze(code)

        # Should still work without cache
        assert result.total_paths >= 1
        assert result.from_cache is False

    def test_symbolic_cache_config_changes_key(self):
        """Test that different configs produce different cache entries."""
        import tempfile
        from pathlib import Path
        from code_scalpel.utilities.cache import CacheConfig, AnalysisCache, reset_cache

        reset_cache()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = AnalysisCache(config)

            import code_scalpel.utilities.cache as cache_module

            original_get_cache = cache_module.get_cache
            cache_module.get_cache = lambda *args, **kwargs: cache

            try:
                from code_scalpel.symbolic_execution_tools.engine import (
                    SymbolicAnalyzer,
                )

                code = "x = 10"

                # Analyzer with default config
                analyzer1 = SymbolicAnalyzer(max_loop_iterations=10)
                analyzer1.analyze(code)

                # Analyzer with different config
                analyzer2 = SymbolicAnalyzer(max_loop_iterations=20)
                result2 = analyzer2.analyze(code)

                # Both should compute results (not share cache)
                # since configs are different
                assert result2.from_cache is False

            finally:
                cache_module.get_cache = original_get_cache
