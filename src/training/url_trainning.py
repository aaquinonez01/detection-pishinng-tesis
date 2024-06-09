import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from features.url.utils import URL_FEATURE_COLUMNS, URL_PROCESS_DATASET_PATH, MODEL_PATH


class UrlModelTrainer:
    def __init__(self, preprocessed_csv_path, model_path):
        self.preprocessed_csv_path = preprocessed_csv_path
        self.model_path = model_path

    def load_dataset(self):
        self.dataset = pd.read_csv(self.preprocessed_csv_path)
        self.X = self.dataset[URL_FEATURE_COLUMNS[:-1]]  # Excluye la columna 'result'
        self.X = self.dataset.scale(self.X)
        self.y = self.dataset["result"]

    def train_model(self):
        self.model = RandomForestClassifier(n_jobs=-1)
        self.model.fit(self.X, self.y)

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        print(f"Modelo guardado en {self.model_path}")
