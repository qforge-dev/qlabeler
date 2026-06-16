FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel setuptools && \
    pip install fastapi boto3 pydub python-multipart "uvicorn[standard]" audioop-lts

COPY services/ /app/services/

EXPOSE 8000

CMD ["uvicorn", "services.pipeline_app:app", "--host", "0.0.0.0", "--port", "8000"]
