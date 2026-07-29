FROM python:3.11-slim

WORKDIR /app

# System deps (ffmpeg for whisper/audio)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health').read()" || exit 1
CMD ["gunicorn", "--bind=0.0.0.0:8000", "--workers=2", "--timeout=120", "wsgi:app"]
