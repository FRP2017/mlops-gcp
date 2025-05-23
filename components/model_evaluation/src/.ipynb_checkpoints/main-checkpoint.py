import argparse
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score
from kubeflow.v2.components import Metrics

def model_evaluation(preprocessed_test_data_path: str, trained_model_path: str,
                     metrics_output_path: str, accuracy_threshold: float,
                     f1_threshold: float):
    """
    Evalúa el modelo entrenado y registra las métricas.
    """
    X_test = pd.read_csv(os.path.join(preprocessed_test_data_path, 'X_test_scaled.csv'))
    y_test = pd.read_csv(os.path.join(preprocessed_test_data_path, 'y_test.csv'))
    model = joblib.load(os.path.join(trained_model_path, 'model.joblib'))

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1_micro = f1_score(y_test, predictions, average='micro') # Micro average para multiclass

    # Registrar métricas para Vertex AI Pipelines UI y Vertex ML Metadata
    metrics = Metrics(name="iris_model_metrics")
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("f1_micro", f1_micro)

    print(f"Accuracy: {accuracy}")
    print(f"F1-score (micro): {f1_micro}")

    # Lógica condicional para "bless" el modelo
    if accuracy >= accuracy_threshold and f1_micro >= f1_threshold:
        with open(os.path.join(metrics_output_path, 'blessed'), 'w') as f:
            f.write('True')
        print("Modelo aprobado para despliegue.")
    else:
        with open(os.path.join(metrics_output_path, 'blessed'), 'w') as f:
            f.write('False')
        print("Modelo NO aprobado para despliegue. No cumple los umbrales de rendimiento.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_test_data_path', type=str)
    parser.add_argument('--trained_model_path', type=str)
    parser.add_argument('--metrics_output_path', type=str)
    parser.add_argument('--accuracy_threshold', type=float)
    parser.add_argument('--f1_threshold', type=float)
    args = parser.parse_args()

    model_evaluation(args.preprocessed_test_data_path, args.trained_model_path,
                     args.metrics_output_path, args.accuracy_threshold,
                     args.f1_threshold)