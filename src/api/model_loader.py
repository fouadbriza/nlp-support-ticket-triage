import os
import joblib
import numpy as np

MODELS_DIR = "models"
MODEL_FILES = {
    "logistic_regression": "logistic_regression_pipeline.joblib",
    "decision_tree": "decision_tree_pipeline.joblib",
    "random_forest": "random_forest_pipeline.joblib",
    "xgboost": "xgboost_pipeline.joblib",
}

class ModelManager:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.models = {}
        self.load_all_models()

    def load_all_models(self):
        for key, fname in MODEL_FILES.items():
            path = os.path.join(self.models_dir, fname)
            if os.path.exists(path):
                self.models[key] = joblib.load(path)
                print(f"[ModelManager] Loaded {key} from {path}")
            else:
                print(f"[ModelManager] Warning: Artifact {path} not found.")

    def predict(self, text: str, model_name: str = "logistic_regression") -> tuple[str, float]:
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not loaded. Available: {list(self.models.keys())}")

        artifact = self.models[model_name]

        # Handle XGBoost bundle (contains pipeline + label encoder)
        if model_name == "xgboost":
            pipeline = artifact['pipeline']
            le = artifact['label_encoder']
            probs = pipeline.predict_proba([text])[0]
            best_idx = int(np.argmax(probs))
            category = str(le.inverse_transform([best_idx])[0])
            confidence = float(probs[best_idx])
            return category, round(confidence, 4)

        # Standard scikit-learn pipelines
        pipeline = artifact
        probs = pipeline.predict_proba([text])[0]
        best_idx = int(np.argmax(probs))
        category = str(pipeline.classes_[best_idx])
        confidence = float(probs[best_idx])
        return category, round(confidence, 4)

    @property
    def available_models(self) -> list[str]:
        return list(self.models.keys())
