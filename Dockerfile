FROM python:3.10-slim-bookworm

# No .pyc files to carry, and logs that reach the container's log stream
# as they are written rather than when a buffer happens to flush.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Pinned runtime dependencies, installed before any source is copied so the
# layer is reused on every build that does not change them.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The metadata the build backend reads, then the package itself. Installed
# regularly rather than editably, and without resolving dependencies again:
# requirements.txt already pinned them.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps .

# The trained artifact, copied in as it is. Nothing here trains a model.
COPY models/churn_model.joblib ./models/churn_model.joblib

# An absolute path, so the artifact resolves wherever the process is run
# from, and the version the application itself defaults to: this deployment
# does not name one, and a plausible-looking stamp would misattribute it.
ENV CHURNER_MODEL_PATH=/app/models/churn_model.joblib \
    CHURNER_MODEL_VERSION=unversioned

# Serve as a non-root user. The application and the model stay root-owned
# and world-readable, so the process can read them but not rewrite them.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Readiness, asked over the port the application already serves, using the
# interpreter that is already in the image. A non-2xx response or a refused
# connection raises, which is the non-zero exit the check reports as
# unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/readiness', timeout=4).read()"]

CMD ["uvicorn", "churner.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
