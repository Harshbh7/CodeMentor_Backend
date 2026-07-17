# ==============================================================
# CodeMentor AI - Dockerfile
# ==============================================================
# Multi-stage build:
# Stage 1 (builder): Install dependencies in an isolated layer.
# Stage 2 (runtime): Copy only what's needed — smaller final image.
#
# Why multi-stage?
# - Keeps the production image lean (no build tools, no pip cache).
# - Improves security surface area.
# - Faster deployment due to smaller image size.
# ==============================================================

# ---- Stage 1: Builder ----
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /build

# Install system dependencies needed to compile certain Python packages
# (e.g., psycopg2 needs libpq-dev, cryptography needs gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching: only re-run pip install if requirements change)
COPY requirements.txt .

# Install Python dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: Runtime ----
FROM python:3.12-slim AS runtime

# Create a non-root user for security
# Running as root inside containers is a security anti-pattern
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Install only runtime system dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appgroup . .

# Create required directories and set ownership
RUN mkdir -p /app/uploads /app/chroma_data /app/logs && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check: Docker will call this to determine container health
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8000/api/v1/health || exit 1

# Default command: run with Uvicorn
# In production, increase --workers based on CPU count (2 * CPUs + 1 is a good rule)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
