# ============================================================================
# EKA backend — FastAPI service container.
#
# Build context is backend/ (see docker-compose.yml: `build: ./backend`).
# NOTE: ml/models/ranker/eka_ranker.txt lives OUTSIDE this build context (it's
# one directory up, in ../ml/models/), so it is intentionally NOT copied into
# this image. docker-compose.yml bind-mounts ./ml/models into the running
# container at /models and sets RANKER_MODEL_PATH to point there — see the
# comment in docker-compose.yml for why that's cleaner than baking the model
# into the image. If you build/run this Dockerfile standalone (no compose),
# either bind-mount that directory yourself at the same path or accept that
# the ranker service will fall back to its heuristic scorer.
# ============================================================================

FROM python:3.11-slim

# ---------------------------------------------------------------- OS deps
# asyncpg, lightgbm, and numpy ship prebuilt wheels for slim Debian, so no
# compiler toolchain is normally required. build-essential is installed only
# as a defensive fallback for the rare case pip has to build a wheel from
# source (e.g. a version bump that drops a prebuilt wheel for this
# platform), then removed in the same layer so it never ends up in the final
# image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Copy requirements first so `pip install` is its own Docker layer — code
# changes elsewhere in backend/ won't force a reinstall of every dependency.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now remove the build toolchain — it was only needed for the pip install
# above, not for running the app.
RUN apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Now copy the rest of the application code.
COPY . .

# ------------------------------------------------------------- non-root user
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin eka \
    && chown -R eka:eka /app
USER eka

EXPOSE 8000

# Render/Docker health checks both hit the same endpoint the app already
# exposes for monitoring (see infra/uptimerobot_note.txt). start-period gives
# the lifespan startup (DB connect, Qdrant collection check) time to finish
# before the first health probe counts against it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
