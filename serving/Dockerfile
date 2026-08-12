# ============================================================================
# EKA serving/ — standalone classifier microservices container.
#
# One image, three possible processes: ranker_serve.py (port 8005, cheap,
# numpy+lightgbm only), complexity_serve.py (port 8004) and sentiment_serve.py
# (port 8007) (both need torch+transformers). Which one runs is decided by
# CMD/the container's command override — see docker-compose.yml's
# `classifiers` profile for examples of overriding it per-service.
#
# Build context is serving/ (see docker-compose.yml: `build: ./serving`).
# ============================================================================

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# build-essential covers the rare case pip needs to compile something from
# source on this platform; removed again after install so it isn't shipped.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so dependency installs are their own cached layer.
COPY requirements.txt .

# --extra-index-url (not --index-url) is deliberate: PyPI is still checked
# for fastapi/uvicorn/pydantic/numpy/lightgbm, while the PyTorch CPU index is
# ALSO checked and is where pip resolves the torch==2.3.0 pin to its CPU-only
# wheel (no CUDA libs, hundreds of MB smaller than the default PyPI torch
# build). Using --index-url instead would make pip look ONLY at the PyTorch
# index and fail to find the non-torch packages.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

RUN apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin eka \
    && chown -R eka:eka /app
USER eka

# ranker_serve.py's default port; complexity/sentiment default to 8004/8007
# respectively (see their own PORT env handling) — override with `-p` on
# `docker run` or the `ports:` mapping in docker-compose.yml to match
# whichever script CMD actually launches.
EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os,urllib.request,sys; port=os.environ.get('PORT', os.environ.get('RANKER_PORT', 8005)); sys.exit(0 if urllib.request.urlopen(f'http://localhost:{port}/health', timeout=4).status == 200 else 1)"

# Default: the cheap ranker service (no torch needed to be useful — it falls
# back to a heuristic scorer if eka_ranker.txt isn't mounted, same idea as
# the backend's classifier fallbacks). Override at `docker run` time, e.g.:
#   docker run eka-serving uvicorn complexity_serve:app --host 0.0.0.0 --port 8004
#   docker run eka-serving uvicorn sentiment_serve:app --host 0.0.0.0 --port 8007
CMD ["uvicorn", "ranker_serve:app", "--host", "0.0.0.0", "--port", "8005"]
