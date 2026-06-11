import tensorflow as tf
import json
import os

class PersonCNNModel:
    def __init__(self, model_path: str, json_path: str):
        self.model_path = model_path
        self.json_path = json_path
        self.model = self._load_model()
        self.categories = self._load_categories()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        return tf.keras.models.load_model(self.model_path)

    def _load_categories(self):
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON labels not found at {self.json_path}")
        with open(self.json_path, 'r') as f:
            data = json.load(f)
            return data['class_names']