# Python 3.12 rather than 3.14: every dependency here (numpy, scipy, shapely,
# trimesh, lxml, mapbox-earcut) ships a cp312 manylinux wheel, so the image
# builds with no compiler and no source builds. Same reasoning as the wheel
# rule in CLAUDE.md.
FROM python:3.12-slim

# Keeps the container's stdout unbuffered so `docker logs` shows the startup
# line immediately, and stops pip caching a layer we throw away.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency layer first: install against a stub package so that editing source
# does not re-download numpy and scipy on every rebuild.
COPY pyproject.toml README.md ./
COPY src/rsupport/__init__.py src/rsupport/
RUN pip install --no-compile .

# Then the real source over the top. --force-reinstall so pip definitely
# replaces the stub, --no-deps so it does not revisit the layer above.
COPY src/ src/
RUN pip install --no-compile --no-deps --force-reinstall .

# Uploads land in a tempdir under the app user; nothing is written to /app.
RUN useradd --create-home --uid 10001 rsupport
USER rsupport

EXPOSE 8000

# Cheapest real route: it exercises the app, not just the socket.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/presets', timeout=4)"

# 0.0.0.0 so the published port reaches it; --no-browser because there is no
# browser in here to open. This is what makes the app live for as long as the
# container does.
CMD ["python", "-m", "rsupport.web", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
