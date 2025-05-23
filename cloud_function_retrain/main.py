import base64
import json
import os
import logging
from google.cloud import aiplatform

# Configuración del proyecto
PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'mlops-third')
LOCATION = os.environ.get('GCP_REGION', 'us-central1')
PIPELINE_ROOT = os.environ.get('PIPELINE_ROOT', 'gs://mlops-third-artifacts-bucket/iris-pipelines')
PIPELINE_TEMPLATE_PATH = os.environ.get('PIPELINE_TEMPLATE_PATH', 'gs://mlops-third-artifacts-bucket/iris-pipelines/iris_mlops_pipeline.json')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def trigger_retrain_pipeline(event, context):
    """
    Cloud Function activada por un mensaje de Pub/Sub (una alerta de monitoreo).
    Desencadena la pipeline de reentrenamiento de MLOps.
    """
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    logger.info(f"Mensaje de Pub/Sub recibido: {pubsub_message}")

    try:
        # Aquí puedes parsear el mensaje para extraer detalles de la alerta si es necesario.
        # Para simplificar, asumimos que cualquier alerta de monitoreo dispara el reentrenamiento.
        message_data = json.loads(pubsub_message)
        logger.info(f"Alerta de monitoreo recibida para: {message_data.get('model_display_name', 'Modelo Desconocido')}")

    except json.JSONDecodeError:
        logger.warning("No se pudo decodificar el mensaje JSON. Procediendo con el reentrenamiento.")

    # Inicializar cliente de Vertex AI
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    try:
        # Obtener una referencia a la pipeline compilada
        pipeline_job = aiplatform.PipelineJob(
            display_name=f'mlops-third-retrain-triggered-{context.event_id}',
            template_path=PIPELINE_TEMPLATE_PATH,
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False, # Reentrenamiento no debe usar cache
            parameter_values={
                'project_id': PROJECT_ID,
                'region': LOCATION,
                'bq_dataset_name': 'mlops_experiment',
                'bq_training_table_name': 'iris_training_data',
                'bq_prediction_table_name': 'iris_prediction_data',
                'accuracy_threshold': 0.95, # Puedes ajustar estos umbrales
                'f1_threshold': 0.90,
                'model_display_name': 'mlops-third-iris-model',
                'batch_predict_output_uri': 'bq://mlops-third/mlops_experiment/iris_predictions',
                'model_monitoring_display_name': 'mlops-third-iris-monitoring',
                'drift_email_recipients': 'your-email@example.com', # Reemplaza con tu email
                'log_sink_pubsub_topic': f'projects/{PROJECT_ID}/topics/mlops-third-retrain-topic',
                'model_monitoring_bq_dataset': 'mlops_experiment_monitoring'
            }
        )
        pipeline_job.submit()
        logger.info(f"Pipeline de reentrenamiento {pipeline_job.name} iniciada exitosamente.")

    except Exception as e:
        logger.error(f"Error al iniciar la pipeline de reentrenamiento: {e}")
        raise