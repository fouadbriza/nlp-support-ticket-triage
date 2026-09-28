import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")

def train_and_evaluate():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train_raw = train_df['text']
    y_train = train_df['label']
    
    X_test_raw = test_df['text']
    y_test = test_df['label']
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    print(f"Feature matrix shape (Rows, Word Features): {X_train.shape}")
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    print("Training Logistic Regression...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nLogistic Regression Accuracy: {acc * 100:.2f}%")
    print(f"Baseline was: 20.66%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_and_evaluate()
