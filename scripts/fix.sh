#!/bin/bash
# =============================================================================
# Code Scalpel - Auto-Fix Script
# =============================================================================
# Automatically fixes formatting and linting issues
# Run with: ./scripts/fix.sh
# =============================================================================

set -e

echo "=============================================="
echo "🔧 Code Scalpel Auto-Fix"
echo "=============================================="
echo ""

# Check we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from project root (where pyproject.toml is)"
    exit 1
fi

echo "🔍 Step 1/2: Formatting with Black..."
echo "----------------------------------------------"
black src tests examples
echo "✅ Formatting complete"
echo ""

echo "🧹 Step 2/2: Fixing linting issues with Ruff..."
echo "----------------------------------------------"
ruff check src tests examples --fix
echo "✅ Linting fixes applied"
echo ""

echo "=============================================="
echo "✅ AUTO-FIX COMPLETE"
echo "=============================================="
echo ""
echo "Now run './scripts/verify.sh' to confirm all checks pass."
echo ""
