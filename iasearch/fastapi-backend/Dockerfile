# Prompt TD2-1: write a Dockerfile for Fastapi with uvicorn from uv docker image
# python 3.12 trixie, using uv sync with pyproject.toml and uv.lock, start uvicorn with uv run
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY main.py rerank.py ./

EXPOSE 8000 5678

# TD2-3: CMD exacte du cours (debugpy + uvicorn)
CMD ["uv", "run", "-m", "debugpy", "--listen", "0.0.0.0:5678", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
