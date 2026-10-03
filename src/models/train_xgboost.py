import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_pipeline.joblib")

def train_and_evaluate():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train_raw = train_df['text']
    X_test_raw = test_df['text']

    le = LabelEncoder()
    y_train = le.fit_transform(train_df['label'])
    y_test = le.transform(test_df['label'])

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                                     random_state=42, eval_metric='mlogloss', verbosity=0))
    ])

    print("Training XGBoost (n_estimators=100, lr=0.1)...")
    pipeline.fit(X_train_raw, y_train)

    acc = accuracy_score(y_test, pipeline.predict(X_test_raw))
    print(f"Test Accuracy: {acc * 100:.2f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({'pipeline': pipeline, 'label_encoder': le}, MODEL_PATH)
    print(f"Saved pipeline + label encoder to: {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
