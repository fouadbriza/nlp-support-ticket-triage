import os
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")

def evaluate_baseline():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = train_df['text']
    y_train = train_df['label']
    
    X_test = test_df['text']
    y_test = test_df['label']
    
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    
    y_pred = dummy.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Baseline Accuracy (Most Frequent Class): {acc * 100:.2f}%")
    
    print("\nBaseline Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    evaluate_baseline()
