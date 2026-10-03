import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_pipeline.joblib")

def train_and_evaluate():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train_raw = train_df['text']
    y_train = train_df['label']
    X_test_raw = test_df['text']
    y_test = test_df['label']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    print("Training Random Forest (n_estimators=100)...")
    pipeline.fit(X_train_raw, y_train)

    acc = accuracy_score(y_test, pipeline.predict(X_test_raw))
    print(f"Test Accuracy: {acc * 100:.2f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
