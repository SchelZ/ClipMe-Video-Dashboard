FROM python:3.12-slim

# System need dep
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create working dir
WORKDIR /app

# Install python dep
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    celery[redis] \
    sqlalchemy \
    python-multipart \
    boto3 \
    ffmpeg-python \
    pydantic \
    requests \
    scenedetect[opencv] \
    opencv-python-headless \
    pyttsx3 \
    google-cloud-speech \
    psycopg2-binary \
    django \
    django-environ \
    git+https://github.com/openai/whisper.git

# Copy project files
COPY opus_starter_bot.py /app/

# Expose API port
EXPOSE 8000

# Start uvicorn server
CMD ["uvicorn", "opus_starter_bot:app", "--host", "0.0.0.0", "--port", "8000"]
