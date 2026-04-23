FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY app/ ./app/
# Worker module is referenced by name in RQ enqueue calls — only __init__ needed here
COPY pipeline/__init__.py ./pipeline/__init__.py

ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python", "-m", "app.main"]
