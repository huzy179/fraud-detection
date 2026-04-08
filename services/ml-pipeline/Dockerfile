FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/

# data/ và models/ được mount từ host qua docker-compose volumes
# nên không cần COPY ở đây

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

CMD ["python", "-c", "from scripts.preprocess import main as pm; from scripts.train import main as tm; pm(); tm()"]
