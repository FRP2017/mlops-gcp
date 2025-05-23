FROM python:3.9-slim-buster

WORKDIR /app

# Instalar dependencias
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    google-cloud-storage \
    joblib

# Copiar el script del predictor
COPY predictor.py .

# Exponer el puerto predeterminado para el servicio de predicción de Vertex AI
# Aunque es Batch Prediction, el contenedor sigue el mismo patrón de servicio
EXPOSE 8080

# Comando para iniciar el predictor
# Vertex AI espera que un servidor HTTP sirva el modelo
# Usamos un servidor simple Flask/gunicorn o similar, pero para Batch,
# la clase CustomPredictor será invocada directamente por el runtime de Vertex AI.
# Esto es solo un placeholder para que el Dockerfile sea válido.
# El predictor de Vertex AI es un wrapper que invoca `load` y `predict` de tu clase.
ENTRYPOINT ["python", "-m", "google.cloud.aiplatform.prediction.model_server"]
# La línea anterior es correcta para la integración de Vertex AI Prediction con una clase CustomPredictor.
# Sin embargo, si quieres probarlo localmente, puedes usar gunicorn/flask
# CMD ["gunicorn", "--bind", "0.0.0.0:8080", "predictor:app"]