# features/url/preprocessing.py

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from extraction import website
from utils import URL_FEATURE_COLUMNS, CSV_PATH, URL_PROCESS_DATASET_PATH


class UrlPreprocessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.website_list = []
        self.list_doubt = []

    def load_dataset(self):
        try:
            self.dataset = pd.read_csv(self.csv_path)
            print("Lectura de archivo CSV exitosa")
        except pd.errors.ParserError as e:
            print(f"Error al leer el archivo CSV: {e}")
            self.dataset = pd.DataFrame()

    def process_url(self, row):
        label = 1 if row["result"] == 0 else -1
        aux = website(row["url"], label)
        aux.getFeatures()
        if aux.doubt == 0:
            return aux.features, None
        else:
            return None, row

    def preprocess(self, max_workers=10):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_url, row): row
                for _, row in self.dataset.iterrows()
            }

            for future in as_completed(futures):
                result, doubt = future.result()
                if result:
                    self.website_list.append(result)
                if doubt:
                    self.list_doubt.append(doubt)

    def save_processed_data(self, output_path):
        dt_finish = pd.DataFrame(self.website_list, columns=URL_FEATURE_COLUMNS)
        dt_finish.to_csv(output_path, index_label="Ord.")
        print(f"Datos preprocesados guardados en {output_path}")
        print("Lista de URL que dieron problemas:")
        print(self.list_doubt)


def main():
    # Ruta al archivo CSV de entrada y salida

    # Inicializa el preprocesador
    url_preprocessor = UrlPreprocessor(CSV_PATH)

    # Cargar el dataset
    url_preprocessor.load_dataset()

    # Preprocesar el dataset
    url_preprocessor.preprocess(max_workers=10)

    # Guardar los datos preprocesados
    url_preprocessor.save_processed_data(URL_PROCESS_DATASET_PATH)


if __name__ == "__main__":
    main()
