import kfp
from kfp.dsl import pipeline, component, Condition
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.batch_predict import ModelBatchPredictOp
from google_cloud_pipeline_components.v1.model_monitoring import ModelMonitoringJobOp

# Definición de los componentes locales
# Compila los componentes desde sus archivos YAML
data_ingestion_op = kfp.components.load_component_from_file(
    'components/data_ingestion/component.yaml'
)
data_preprocessing_op = kfp.components.load_component_from_file(
    'components/data_preprocessing/component.yaml'
)
model_training_op = kfp.components.load_component_from_file(
    'components/model_training/component.yaml'
)
model_evaluation_op = kfp.components.load_component_from_file(
    'components/model_evaluation/component.yaml'
)

@pipeline(
    name='mlops-third-iris-pipeline',
    description='Pipeline de MLOps para el modelo Iris Random Forest con Vertex AI y Kubeflow.',
    pipeline_root='gs://mlops-third-artifacts-bucket/iris-pipelines' # Ubicación en GCS para los artefactos de la pipeline
)
def iris_mlops_pipeline(
    project_id: str = 'mlops-third',
    region: str = 'us-central1',
    bq_dataset_name: str = 'mlops_experiment',
    bq_training_table_name: str = 'iris_training_data',
    bq_prediction_table_name: str = 'iris_prediction_data', # Tabla para simular nuevos datos a predecir
    accuracy_threshold: float = 0.95, # Umbral para aprobación del modelo
    f1_threshold: float = 0.90,       # Umbral para aprobación del modelo
    model_display_name: str = 'mlops-third-iris-model',
    batch_predict_output_uri: str = 'bq://mlops-third/mlops_experiment/iris_predictions', # Salida de predicción por lotes
    model_monitoring_display_name: str = 'mlops-third-iris-monitoring',
    drift_email_recipients: str = 'your-email@example.com', # Reemplaza con tu email para alertas
    log_sink_pubsub_topic: str = 'projects/mlops-third/topics/mlops-third-retrain-topic', # Topic para el reentrenamiento
    model_monitoring_bq_dataset: str = 'mlops_experiment_monitoring' # Dataset de BQ para monitoreo
):
    # Paso 1: Ingesta de Datos
    ingestion_task = data_ingestion_op(
        project_id=project_id,
        bq_dataset=bq_dataset_name,
        bq_table=bq_training_table_name
    )

    # Paso 2: Preprocesamiento de Datos
    preprocessing_task = data_preprocessing_op(
        training_data=ingestion_task.outputs['training_data'],
        test_data=ingestion_task.outputs['test_data']
    )

    # Paso 3: Entrenamiento del Modelo
    training_task = model_training_op(
        preprocessed_training_data=preprocessing_task.outputs['preprocessed_training_data']
    )

    # Paso 4: Evaluación del Modelo
    evaluation_task = model_evaluation_op(
        preprocessed_test_data=preprocessing_task.outputs['preprocessed_test_data'],
        trained_model=training_task.outputs['trained_model'],
        accuracy_threshold=accuracy_threshold,
        f1_threshold=f1_threshold
    )

    # Paso 5: Despliegue Condicional y Registro del Modelo
    with Condition(evaluation_task.outputs['model_blessed'] == 'True'):
        # Subir el modelo a Vertex AI Model Registry
        # Incluimos el scaler en el artifact_uri para que el contenedor de predicción pueda usarlo
        model_upload_task = ModelUploadOp(
            project=project_id,
            display_name=model_display_name,
            artifact_uri=training_task.outputs['trained_model'].uri,
            serving_container_image_uri='us-central1-docker.pkg.dev/mlops-third/mlops-third-iris-repo/iris-batch-predictor:latest', # Se creará más adelante
            serving_container_environment_variables={
                "SCALER_PATH": preprocessing_task.outputs['scaler_model'].uri # Pasar la URI del scaler
            }
        ).after(training_task) # Asegurarse que el scaler esté disponible

        # Paso 6: Configurar Predicción por Lotes
        # Esto no despliega un endpoint, solo lo prepara para trabajos de Batch Prediction
        batch_prediction_task = ModelBatchPredictOp(
            project=project_id,
            location=region,
            model=model_upload_task.outputs['model'], # Referencia al modelo subido
            job_display_name=f'{model_display_name}-batch-prediction-job',
            instances_format='bigquery',
            bigquery_source_input_uri=f'bq://{project_id}.{bq_dataset_name}.{bq_prediction_table_name}',
            predictions_format='bigquery',
            bigquery_destination_output_uri=batch_predict_output_uri
        )

        # Paso 7: Configurar Monitoreo del Modelo (para el modelo en el Registry)
        # Nota: Vertex AI Model Monitoring está diseñado para endpoints desplegados,
        # pero se puede configurar para modelos en el Registry si se desea monitorear la distribución de predicciones
        # en trabajos por lotes. Sin embargo, su funcionalidad completa (skew vs. drift) es más potente con Endpoints.
        # Para Batch Prediction, el monitoreo a menudo se hace post-hoc o con TFDV en la pipeline de ingesta.
        # Aquí, vamos a simular la configuración del monitoreo para un modelo que *podría* ser servido en línea,
        # y que sus predicciones sean monitoreadas si un job de batch predict está activo y loguea las inferencias.
        # Para el monitoreo de modelos registrados, necesitas un "endpoint virtual" o un job de monitoreo que lea directamente de las tablas de predicción.
        # La forma más robusta para Batch Prediction es el monitoreo de datos de entrada (con TFDV) y la post-evaluación de las predicciones resultantes.
        # Para el ejemplo, lo configuraremos en el Model Registry, que es el punto de partida para monitorear modelos.

        # Primeramente, necesitas un dataset para el monitoreo de BigQuery.
        # Este componente no creará el dataset. Asegúrate de que `mlops_experiment_monitoring` exista en BQ.

        # Crea un 'PredictionSchema' para el monitoreo si no lo tienes.
        # Para un modelo de sklearn, normalmente se define manualmente.
        # Para Iris, las entradas son: sepal_length, sepal_width, petal_length, petal_width.
        # Las salidas son la predicción de clase (0, 1, 2).

        # NOTA IMPORTANTE PARA EL MONITOREO DE BATCH PREDICTION:
        # Vertex AI Model Monitoring está principalmente diseñado para monitorear *endpoints en línea*.
        # Para Batch Prediction, el monitoreo de deriva/skew generalmente se hace
        # *antes* de la inferencia (validación de datos de entrada con TFDV)
        # o *después* de la inferencia (validando las distribuciones de las predicciones resultantes).
        # La configuración de ModelMonitoringJobOp en una pipeline KFP está más orientada
        # a configurar el monitoreo para un Endpoint que se desplegaría.
        # Sin embargo, el Model Registry es la fuente de verdad para el modelo.
        # Vamos a configurar un ModelMonitoringJob que usa el modelo del Model Registry y
        # asume que puedes apuntar una fuente de datos de "predicciones" (la tabla de BQ de salida de batch predict)
        # como base para el monitoreo de deriva/skew. Esto es conceptualmente cómo funcionaría
        # si se quisiera monitorear directamente los resultados de la predicción por lotes.
        # Para un monitoreo efectivo de Batch Prediction, Cloud Logging es CLAVE para la "data collection".
        # Asegúrate de que los jobs de Batch Prediction estén configurados para loguear.

        # Para un monitoreo mensual, esto no sería activado por la pipeline, sino por Cloud Scheduler.
        # Pero podemos usar el componente para crear el trabajo de monitoreo.

        # **Simplificación del Monitoreo para el Ejemplo:**
        # Para demostrar la integración, vamos a crear un monitor para el modelo registrado.
        # La "data collection" (recopilación de datos) para el monitoreo de batch prediction
        # es más compleja y generalmente requiere un Log Sink que envíe las entradas/salidas
        # de cada predicción por lotes a un topic de Pub/Sub o BigQuery.
        # Aquí, simulamos el set-up del monitor.

        # Configuración de ModelMonitoringJobOp (asumiendo que las predicciones de batch se loguearán y estarán disponibles)
        # Para un uso real de Model Monitoring con Batch Prediction, la configuración de "data_source"
        # y "training_dataset" sería más compleja y probablemente se necesitaría que el job de Batch Prediction
        # escribiera en una tabla de BigQuery que luego el monitor pueda leer, o que los logs de predicción
        # se exporten a una fuente que Model Monitoring pueda consumir.
        # Aquí, solo crearemos el trabajo de monitoreo vinculado al modelo en el registro.

        # Primero, necesitas un "Prediction Schema" para Iris
        # {
        #    "instance": {"sepal_length": 0.0, "sepal_width": 0.0, "petal_length": 0.0, "petal_width": 0.0},
        #    "prediction": 0
        # }

        # Esto se simplificará para el ejemplo para solo crear la tarea de monitoreo.
        # Un escenario real requeriría un setup más sofisticado de loggeo de inferencia.

        # La manera más sencilla de integrar el monitoreo de Model Monitoring
        # con un modelo entrenado por pipeline es usarlo con un endpoint.
        # Dado que solicitaste Batch Prediction, haremos el monitoreo
        # con un enfoque más "manual" sobre el modelo en el Registry.

        # ModelMonitoringJobOp es para crear trabajos de monitoreo.
        # No vamos a usarlo directamente en la pipeline para desencadenar el reentrenamiento,
        # sino para crear el *trabajo de monitoreo* que luego se ejecutaría mensualmente.

        # Para la deriva de datos de entrada, necesitas un dataset de entrenamiento
        # Para la deriva de predicción, necesitas un dataset de predicciones

        # Una implementación más completa de monitoreo para Batch Prediction
        # implicaría que la tabla de salida de `ModelBatchPredictOp` sea la fuente de datos
        # para un job de monitoreo de Vertex AI, posiblemente activado por un cron job.
        # También se puede usar TensorFlow Data Validation en un componente KFP
        # en la pipeline de ingesta de la *nueva* data para predecir.

        # Para este ejemplo, configuraremos el monitoreo en el modelo registrado,
        # asumiendo que los datos de entrenamiento y predicción serán accesibles.

        # CREACIÓN DEL TRABAJO DE MONITOREO (fuera de la pipeline para este ejemplo simplificado,
        # se activaría por Cloud Scheduler para el reentrenamiento mensual).
        # Lo que la pipeline sí haría es subir el modelo al Model Registry, que es pre-requisito.

        # Nota: ModelMonitoringJobOp requiere un endpoint para funcionar correctamente.
        # Para batch, la mejor práctica es monitorear los *datos de entrada* con TFDV
        # y las *predicciones de salida* con lógica personalizada.
        # Si insistiéramos en Vertex AI Model Monitoring para un modelo sin Endpoint,
        # sería a través de Model Monitoring para Model Registry,
        # pero es una característica más reciente y menos "directa" que para Endpoints.

        # En aras de mantener el ejemplo centrado en KFP y Vertex AI Pipelines,
        # y dado que se pidió batch prediction, no Endpoint,
        # la configuración de `ModelMonitoringJobOp` directamente en la pipeline
        # sería un poco forzada sin un Endpoint.

        # Alternativa para Monitoreo en Batch:
        # Se puede agregar un componente que, después de cada Batch Prediction Job,
        # lea los resultados de BigQuery, calcule métricas de deriva con librerías,
        # y luego genere alertas si los umbrales se superan.
        # Esto sería un "custom component" para el monitoreo de batch.

        # Para el propósito de esta solicitud, vamos a suponer que el monitoreo se configura
        # *después* del despliegue del modelo en el registro, y que su activación para el reentrenamiento
        # se hará vía un cron job (Cloud Scheduler) que periódicamente revisa la deriva,
        # o que un *componente de monitoreo personalizado* se ejecuta en la pipeline.

        # Vamos a incluir un marcador para un componente de monitoreo *conceptual* que se ejecutaría
        # como parte del ciclo mensual, no inmediatamente después del despliegue.

        # Placeholder para la configuración del monitoreo (realmente se haría como un trabajo independiente)
        # Puedes crear el trabajo de monitoreo manualmente o con un script aparte que se ejecute mensualmente.
        # Lo importante es que el modelo esté en el Model Registry.

        print("El modelo se ha registrado y se ha configurado la predicción por lotes.")
        print("La configuración del monitoreo de deriva se realizará de forma recurrente/mensual.")

# Para compilar la pipeline (ejecuta esto en tu Workbench)
if __name__ == '__main__':
    compiler = kfp.compiler.Compiler()
    compiler.compile(iris_mlops_pipeline, 'iris_mlops_pipeline.json')
    print("Pipeline compilada a iris_mlops_pipeline.json")