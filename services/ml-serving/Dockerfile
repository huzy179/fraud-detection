FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Models and data are mounted as volumes in docker-compose
# If running locally (no Docker), also mount these volumes

ENV PYTHONPATH=/app
ENV FRAUD_THRESHOLD=0.5
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]