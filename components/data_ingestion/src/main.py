import argparse
import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
import os

def data_ingestion(project_id: str, bq_dataset: str, bq_table: str,
                   train_output_path: str, test_output_path: str):
    """
    Ingiere datos de BigQuery y los divide en conjuntos de entrenamiento y prueba.
    """
    client = bigquery.Client(project=project_id)
    query = f"SELECT sepal_length, sepal_width, petal_length, petal_width, target FROM `{project_id}.{bq_dataset}.{bq_table}`"
    df = client.query(query).to_dataframe()

    # Dividir datos en entrenamiento y prueba
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Guardar en GCS (Vertex AI Pipelines se encargará de esto automáticamente para artefactos)
    # Por simplicidad, guardamos directamente como CSV
    # Los path son locales dentro del contenedor, KFP maneja la persistencia a GCS
    X_train.to_csv(os.path.join(train_output_path, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(train_output_path, 'y_train.csv'), index=False)
    X_test.to_csv(os.path.join(test_output_path, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(test_output_path, 'y_test.csv'), index=False)

    print(f"Datos de entrenamiento guardados en {train_output_path}")
    print(f"Datos de prueba guardados en {test_output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, help='GCP project ID')
    parser.add_argument('--bq_dataset', type=str, help='BigQuery dataset name')
    parser.add_argument('--bq_table', type=str, help='BigQuery table name')
    parser.add_argument('--train_output_path', type=str, help='Path to store training data')
    parser.add_argument('--test_output_path', type=str, help='Path to store test data')
    args = parser.parse_args()

    data_ingestion(args.project_id, args.bq_dataset, args.bq_table,
                   args.train_output_path, args.test_output_path)