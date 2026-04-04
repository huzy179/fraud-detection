FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY services/ml-serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install ml-pipeline dependencies for training inside container
COPY services/ml-pipeline/requirements.txt /tmp/ml-pipeline-requirements.txt
RUN pip install --no-cache-dir -r /tmp/ml-pipeline-requirements.txt

COPY services/ml-serving/main.py .

ENV PYTHONPATH=/app
ENV FRAUD_THRESHOLD=0.5
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]