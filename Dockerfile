# Build context expects chroma_db/ to exist (ingestion runs offline, index is baked in).
FROM python:3.13-slim

# libcap2-bin: setcap lets non-root python bind port 80 (Cloudflare proxy only
# forwards to a fixed port list; 8501 is not on it, 80 is).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcap2-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

RUN setcap 'cap_net_bind_service=+ep' "$(readlink -f "$(which python3)")" \
    && useradd --create-home app

COPY src/ src/
COPY chroma_db/ chroma_db/

ENV VECTOR_STORE_TYPE=chroma \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1/_stcore/health', timeout=4)"]

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=80", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
