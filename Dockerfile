FROM python:3.9-slim-buster

WORKDIR /app

COPY components /app/components

# Instalar dependencias
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    google-cloud-bigquery \
    google-cloud-aiplatform \
    kfp==2.7.0 \
    joblib

# Directorio de trabajo predeterminado para los scripts
ENV PYTHONPATH=/app/components