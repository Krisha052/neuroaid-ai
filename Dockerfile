FROM python:3.11-slim

WORKDIR /app

# System deps (ffmpeg for whisper/audio)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "src/web/app_streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]
