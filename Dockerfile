FROM ubuntu:24.04 AS builder

# Install Python and required packages
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    curl \
    ca-certificates \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Set environment
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=100 \
    PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv" \
    LOGGER_DIR="/app/logs" \
    SERVER_DEBUG=0 \
    HOST=0.0.0.0 \
    PORT=9540

# Configure Git to use HTTPS instead of SSH
RUN git config --global url."https://".insteadOf git://
RUN git config --global url."https://github.com/".insteadOf git@github.com:

WORKDIR /app

# Install uv
RUN pip3 install --break-system-packages uv

RUN python3 -m pip install --break-system-packages --upgrade "setuptools==65.5.0"

# Add non-root user
RUN useradd --create-home --shell /bin/bash --uid 10001 biocentral-server-user

# Create directories including HuggingFace cache directory
RUN mkdir -p /app/logs /var/log/biocentral-server /app/huggingface_models && \
    chown -R biocentral-server-user:biocentral-server-user /app /var/log/

# Copy only dependency files first to leverage Docker caching
COPY pyproject.toml uv.lock ./
RUN touch README.md

# Optional: Set PyTorch index URL (pass --build-arg UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu for CPU-only)
ARG UV_EXTRA_INDEX_URL
ENV UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL}

# Copy application files (this layer changes frequently)
COPY --chown=biocentral-server-user:biocentral-server-user ./biocentral_server ./biocentral_server

# Final sync to install the project itself (should be fast - just installs the project)
RUN uv sync && rm -rf /root/.cache/uv

# Switch to non-root user
USER biocentral-server-user

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn biocentral_server.main:app --host $HOST --port $PORT"]
