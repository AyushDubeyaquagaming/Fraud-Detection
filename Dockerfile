# ─── Stage 1: Build dependencies ───────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps -e .


# ─── Stage 2: Runtime image ─────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY orchestration/ ./orchestration/
COPY tests/ ./tests/
COPY configs/ ./configs/
COPY pyproject.toml ./
COPY README.md ./
COPY ["ROULET CHEATING DATA.csv", "./"]

# Install the local package (already in site-packages from builder, just register)
RUN pip install --no-cache-dir --no-deps -e .

# Create non-root user and writable volume directories
RUN adduser --disabled-password --gecos "" --uid 1000 fraud && \
    mkdir -p /app/artifacts /app/logs /app/mlruns /app/data_cache && \
    chown -R fraud:fraud /app

COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER fraud

# Named volumes that should be mounted from the host
VOLUME ["/app/artifacts", "/app/logs", "/app/mlruns", "/app/data_cache"]

ENTRYPOINT ["/entrypoint.sh"]
CMD ["train"]
