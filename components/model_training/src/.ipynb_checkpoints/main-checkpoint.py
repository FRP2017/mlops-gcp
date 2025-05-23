import argparse
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

def model_training(preprocessed_training_data_path: str, model_output_path: str):
    """
    Entrena un modelo Random Forest Classifier.
    """
    X_train = pd.read_csv(os.path.join(preprocessed_training_data_path, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(preprocessed_training_data_path, 'y_train.csv'))

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel()) # .values.ravel() para evitar UserWarning

    # Guardar el modelo entrenado
    model_filename = os.path.join(model_output_path, 'model.joblib')
    joblib.dump(model, model_filename)

    print(f"Modelo entrenado y guardado en {model_output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_training_data_path', type=str)
    parser.add_argument('--model_output_path', type=str)
    args = parser.parse_args()

    model_training(args.preprocessed_training_data_path, args.model_output_path)