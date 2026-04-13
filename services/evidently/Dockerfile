FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt || \
    pip install --no-cache-dir --timeout 300 -r requirements.txt || \
    pip install --no-cache-dir --timeout 300 -r requirements.txt

COPY evidently_service.py .

ENV PYTHONPATH=/app
ENV PORT=8002

EXPOSE 8002

CMD ["uvicorn", "evidently_service:app", "--host", "0.0.0.0", "--port", "8002"]