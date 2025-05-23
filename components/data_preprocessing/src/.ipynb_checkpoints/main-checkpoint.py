import argparse
import pandas as pd
import numpy as np
import os
import joblib # Para guardar el scaler

def data_preprocessing(training_data_path: str, test_data_path: str,
                       preprocessed_training_data_path: str,
                       preprocessed_test_data_path: str,
                       scaler_output_path: str):
    """
    Realiza preprocesamiento de datos (escalado) en los conjuntos de entrenamiento y prueba.
    """
    X_train = pd.read_csv(os.path.join(training_data_path, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(training_data_path, 'y_train.csv'))
    X_test = pd.read_csv(os.path.join(test_data_path, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(test_data_path, 'y_test.csv'))

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convertir de nuevo a DataFrame para mantener los nombres de columna
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Guardar datos preprocesados
    X_train_scaled_df.to_csv(os.path.join(preprocessed_training_data_path, 'X_train_scaled.csv'), index=False)
    y_train.to_csv(os.path.join(preprocessed_training_data_path, 'y_train.csv'), index=False) # y_train no cambia
    X_test_scaled_df.to_csv(os.path.join(preprocessed_test_data_path, 'X_test_scaled.csv'), index=False)
    y_test.to_csv(os.path.join(preprocessed_test_data_path, 'y_test.csv'), index=False) # y_test no cambia

    # Guardar el scaler para usarlo en la inferencia
    joblib.dump(scaler, os.path.join(scaler_output_path, 'scaler.joblib'))

    print(f"Datos preprocesados guardados en {preprocessed_training_data_path} y {preprocessed_test_data_path}")
    print(f"Scaler guardado en {scaler_output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_path', type=str)
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--preprocessed_training_data_path', type=str)
    parser.add_argument('--preprocessed_test_data_path', type=str)
    parser.add_argument('--scaler_output_path', type=str)
    args = parser.parse_args()

    data_preprocessing(args.training_data_path, args.test_data_path,
                       args.preprocessed_training_data_path,
                       args.preprocessed_test_data_path,
                       args.scaler_output_path)