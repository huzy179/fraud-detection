FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy data and scripts
COPY scripts/ ./scripts/
COPY data/ ./data/

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Training runs once on container start
CMD ["python", "-c", "from scripts.preprocess import main as pm; from scripts.train import main as tm; pm(); tm()"]
