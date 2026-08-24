# Dockerfile for the Predictive Maintenance Demonstrator web app (story H1).
#
# Builds a single self-contained image that serves the Streamlit app with
# `docker run -p 8501:8501 <image>` -- no volume mounts or extra setup
# required. Simulation scenario CSVs (core/simulation_loader.py) are
# generated on first boot inside the container's own filesystem the same
# way they are in local dev (idempotently, from app.py's main()).
#
# Layer-caching note: requirements.txt is copied and installed *before*
# the rest of the application source is copied, so code-only changes
# (no dependency changes) don't invalidate the (slow) pip install layer.

FROM python:3.11-slim

# Prevent Python from writing .pyc files / buffering stdout so that
# `docker logs` shows output immediately (useful for debugging startup).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- Dependency layer (cached unless requirements.txt changes) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application layer ---
# Only production-runtime code/config is copied in; dev-only artifacts
# (tests, .venv, __pycache__, requirements-dev.txt, docs) are excluded
# via .dockerignore.
COPY . .

# assets/simulations/ must be writable at runtime: generate_scenarios()
# (called from app.py's main() on every boot) writes the three scenario
# CSVs there the first time the app runs inside the container, since the
# .dockerignore excludes any pre-generated CSVs. The COPY above already
# creates the directory (it's part of the repo tree); this line just
# guarantees it exists even if .dockerignore excluded its contents
# entirely, and that it's writable by whichever user runs the container.
RUN mkdir -p /app/assets/simulations

EXPOSE 8501

# --server.headless=true avoids Streamlit's interactive first-run email
# prompt (which would otherwise hang `docker run` since there is no TTY
# to answer it). --server.address/--server.port are set explicitly
# rather than relying on defaults so the container is reachable from
# the host via the published port regardless of Streamlit's own default
# config. --browser.gatherUsageStats=false additionally suppresses the
# telemetry opt-in prompt/network call on first run.
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.address=0.0.0.0", \
    "--server.port=8501", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
