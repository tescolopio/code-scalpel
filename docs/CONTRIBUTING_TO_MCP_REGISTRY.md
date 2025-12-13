# Contributing Code Scalpel to the MCP Registry

This guide outlines the steps to publish **Code Scalpel** to the official [Model Context Protocol (MCP) Registry](https://registry.modelcontextprotocol.io/).

## Prerequisites

1.  **GitHub Account**: You need a GitHub account to authenticate with the registry.
2.  **PyPI Account**: You need to publish the `code-scalpel` package to [PyPI](https://pypi.org/).
3.  **mcp-publisher**: The CLI tool for publishing to the registry.

## Step 1: Prepare the Package

### 1. Update Metadata
Ensure `pyproject.toml` has the correct version and description.
Current version: `1.1.1`

### 2. Add Verification Tag to README
The MCP Registry verifies ownership by checking for a specific comment in your `README.md` (which becomes the PyPI description).

We have added a placeholder to the end of `README.md`:
```markdown
<!-- mcp-name: io.github.tescolopio/code-scalpel -->
```

**ACTION REQUIRED:**
1.  Open `README.md`.
2.  Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username (lowercase).
3.  Save the file.

### 3. Configure `server.json`
We have created a `server.json` file in the root directory.

**ACTION REQUIRED:**
1.  Open `server.json`.
2.  Replace `YOUR_GITHUB_USERNAME` in the `"name"` field with your actual GitHub username.
    *   Example: `"name": "io.github.tescolopio/code-scalpel"`
3.  Ensure the `"version"` matches your `pyproject.toml` version.

## Step 2: Publish to PyPI

The MCP Registry does not host code, only metadata. It points to the PyPI package.

1.  **Build the package:**
    ```bash
    pip install build
    python -m build
    ```

2.  **Publish to PyPI:**
    ```bash
    pip install twine
    twine upload dist/*
    ```
    *Note: Ensure you have updated the README on PyPI by publishing a new version.*

## Step 3: Publish to MCP Registry

1.  **Install `mcp-publisher`:**
    ```bash
    # macOS / Linux
    brew install mcp-publisher
    # OR download from https://github.com/modelcontextprotocol/registry/releases
    ```

2.  **Login:**
    ```bash
    mcp-publisher login github
    ```
    Follow the authentication prompts.

3.  **Publish:**
    ```bash
    mcp-publisher publish
    ```

## Verification

After publishing, you can verify your server is listed by searching the registry API:
```bash
curl "https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.YOUR_USERNAME/code-scalpel"
```
