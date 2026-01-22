# -------------------------------------------------------------------
# STAGE 1: BUILDER
# -------------------------------------------------------------------
FROM python:3.11-slim as builder

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for building Python packages
# (e.g., gcc, python3-dev). We clean up afterwards to keep this stage clean,
# though it matters less since this stage is discarded.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to isolate dependencies
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy only requirements first to leverage Docker Layer Caching
COPY src/frontend/requirements.txt .

# Install dependencies into the virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------------------
# STAGE 2: RUNTIME
# -------------------------------------------------------------------
FROM python:3.11-slim as runtime

WORKDIR /app

# Create a non-root user and group for security
# UID 1000 is standard for the first non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1000 appuser

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Update PATH to prioritize the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

# Copy the application source code
# We change ownership to the non-root user immediately
COPY --chown=appuser:appgroup src/frontend/ ./src/frontend/

# Switch to the non-root user
USER appuser

# Expose the port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "src/frontend/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]