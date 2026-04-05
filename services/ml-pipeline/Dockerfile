FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY services/ml-pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir evidently[ui] sqlalchemy psycopg2-binary

COPY services/ml-pipeline/scripts/ ./scripts/
COPY data/ ./data/
COPY models/ ./models/

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

CMD ["python", "-c", "from scripts.preprocess import main as pm; from scripts.train import main as tm; pm(); tm()"]
