# experimenting with different C values for regularization

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")

def experiment_regularization():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train_raw = train_df['text']
    y_train = train_df['label']
    X_test_raw = test_df['text']
    y_test = test_df['label']

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    C_values = [0.001, 0.01, 0.1, 1.0, 10, 100]

    print(f"{'C':>8} | {'Train Acc':>10} | {'Test Acc':>10} | {'Gap':>8}")
    print("-" * 45)

    for C in C_values:
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        gap = train_acc - test_acc

        print(f"{C:>8} | {train_acc*100:>9.2f}% | {test_acc*100:>9.2f}% | {gap*100:>7.2f}%")

if __name__ == "__main__":
    experiment_regularization()
