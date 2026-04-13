#!/bin/bash
# Airflow init script — chạy trước khi webserver/scheduler start
# Chờ DB ready + tạo HTTP connection cho Evidently

set -e

echo "Waiting for Airflow database..."
until airflow info > /dev/null 2>&1; do
    echo "  DB not ready, waiting..."
    sleep 5
done
echo "DB ready."

echo "Creating Airflow connections..."

# Evidently microservice HTTP connection
airflow connections add \
    'evidently_service' \
    --conn-type http \
    --conn-host http://evidently-service \
    --conn-port 8002 \
    --conn-description "Evidently AI drift detection microservice" \
    2>/dev/null && echo "  evidently_service connection created" \
    || echo "  evidently_service connection already exists (skipping)"

echo "Airflow init done. Starting $@..."
exec airflow "$@"