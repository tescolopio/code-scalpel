# Code Scalpel MCP Server Dockerfile
# Multi-stage build for smaller image size
#
# This container runs the MCP-compliant server using streamable-http transport.
# For local development, use stdio transport directly (no container needed).

FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Install the package
COPY . .
RUN pip install --no-cache-dir --user -e .

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Expose the MCP server port
EXPOSE 8080

# Health check for HTTP transport
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/sse -H "Accept: text/event-stream" || exit 1

# Default project root inside container
ENV SCALPEL_ROOT=/app/code

# Run the MCP server with HTTP transport
# Host 0.0.0.0 is required for Docker networking
# --allow-lan disables host validation for external access
# --root points to the mounted code directory
CMD ["code-scalpel", "mcp", "--http", "--host", "0.0.0.0", "--port", "8080", "--root", "/app/code", "--allow-lan"]
