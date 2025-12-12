#!/usr/bin/env python3
"""
Quick verification script to confirm deployment.
Run this to verify Code Scalpel v1.2.0 is properly deployed.
"""

import sys
import subprocess


def check_installation():
    """Verify package installation."""
    print("=" * 60)
    print("Code Scalpel v1.2.0 Deployment Verification")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ["pip", "show", "code-scalpel"],
            capture_output=True,
            text=True,
            check=True
        )
        
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                version = line.split(':')[1].strip()
                print(f"\n✓ Package installed: code-scalpel v{version}")
                if version == "1.2.0":
                    print("  ✓ Version correct")
                else:
                    print(f"  ✗ Expected v1.2.0, got v{version}")
                    return False
                break
        
    except subprocess.CalledProcessError:
        print("\n✗ Package not installed")
        return False
    
    return True


def check_modules():
    """Verify critical modules exist."""
    print("\n" + "-" * 60)
    print("Checking critical modules...")
    print("-" * 60)
    
    try:
        from code_scalpel.surgical_extractor import SurgicalExtractor
        print("✓ surgical_extractor module")
        
        from code_scalpel.project_crawler import ProjectCrawler
        print("✓ project_crawler module")
        
        from code_scalpel.mcp.server import mcp
        print("✓ mcp.server module")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def check_cli():
    """Verify CLI is available."""
    print("\n" + "-" * 60)
    print("Checking CLI availability...")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            ["code-scalpel", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if "code-scalpel" in result.stdout and "mcp" in result.stdout:
            print("✓ CLI command available")
            print("✓ MCP subcommand registered")
            return True
        else:
            print("✗ CLI not properly configured")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"✗ CLI check failed: {e}")
        return False


def main():
    """Run all verification checks."""
    checks = [
        check_installation(),
        check_modules(),
        check_cli(),
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ DEPLOYMENT VERIFIED")
        print("=" * 60)
        print("\nCode Scalpel v1.2.0 is ready for E2E testing!")
        print("\nTo start the MCP server:")
        print("  $ code-scalpel mcp")
        print("\nFor HTTP transport:")
        print("  $ code-scalpel mcp --transport streamable-http --port 8080")
        return 0
    else:
        print("❌ DEPLOYMENT VERIFICATION FAILED")
        print("=" * 60)
        print("\nSome checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
