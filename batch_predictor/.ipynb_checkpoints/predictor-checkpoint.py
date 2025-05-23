import os
import joblib
import pandas as pd
from google.cloud import storage # Para descargar el scaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomPredictor:
    """
    Clase de predictor para el modelo de Random Forest de Iris,
    incluyendo la carga y aplicación del scaler.
    """

    def __init__(self):
        self._model = None
        self._scaler = None

    def load(self, model_dir: str):
        """
        Carga el modelo y el scaler del directorio proporcionado.
        model_dir será gs://<bucket>/<caminodelmodelo>/
        """
        logger.info(f"Cargando modelo desde: {model_dir}")
        model_path = os.path.join(model_dir, 'model.joblib')
        scaler_path = os.environ.get("SCALER_PATH") # URI GCS del scaler desde env variable

        if not os.path.exists(model_path):
             # Si no está directamente, puede ser que el SDK lo haya movido al subdirectorio "model"
            if os.path.exists(os.path.join(model_dir, 'model', 'model.joblib')):
                model_path = os.path.join(model_dir, 'model', 'model.joblib')
            else:
                raise FileNotFoundError(f"model.joblib no encontrado en {model_dir} o {os.path.join(model_dir, 'model')}")

        self._model = joblib.load(model_path)
        logger.info("Modelo cargado exitosamente.")

        if scaler_path:
            logger.info(f"Cargando scaler desde: {scaler_path}")
            # Descargar scaler de GCS
            bucket_name = scaler_path.split('gs://')[1].split('/')[0]
            blob_name = '/'.join(scaler_path.split('gs://')[1].split('/')[1:])
            blob_name_full = os.path.join(blob_name, 'scaler.joblib')

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name_full)
            local_scaler_path = "/tmp/scaler.joblib" # Guardar temporalmente
            blob.download_to_filename(local_scaler_path)
            self._scaler = joblib.load(local_scaler_path)
            logger.info("Scaler cargado exitosamente.")
        else:
            logger.warning("No se encontró SCALER_PATH en las variables de entorno. Las predicciones no serán escaladas.")


    def predict(self, instances: list) -> list:
        """
        Realiza predicciones sobre una lista de instancias.
        """
        logger.info(f"Recibidas {len(instances)} instancias para predicción.")
        # Las instancias de Batch Prediction son un diccionario
        # Convertir a DataFrame para asegurar el orden de las columnas y aplicar scaler
        df_instances = pd.DataFrame(instances)
        
        # Asegurar el orden de las columnas como se entrenó el modelo
        expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        df_instances = df_instances[expected_columns]

        if self._scaler:
            instances_scaled = self._scaler.transform(df_instances)
            logger.info("Instancias escaladas.")
        else:
            instances_scaled = df_instances.values # Convertir a numpy array

        predictions = self._model.predict(instances_scaled)
        logger.info("Predicciones realizadas.")

        # Convertir a lista de int para la salida JSON
        return predictions.tolist()