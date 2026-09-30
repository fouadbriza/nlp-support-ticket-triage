import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_pipeline.joblib")

def train_and_evaluate():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train_raw = train_df['text']
    y_train = train_df['label']
    X_test_raw = test_df['text']
    y_test = test_df['label']
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('classifier', LogisticRegression(C=10.0, max_iter=1000, random_state=42))
    ])
    
    print("Training Pipeline on raw text...")
    pipeline.fit(X_train_raw, y_train)
    
    y_pred = pipeline.predict(X_test_raw)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nPipeline Test Accuracy (C=10): {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved complete pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
