import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")

def experiment_decision_tree():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train_raw = train_df['text']
    y_train = train_df['label']
    X_test_raw = test_df['text']
    y_test = test_df['label']

    depth_values = [5, 10, 20, 50, None]

    print(f"{'max_depth':>10} | {'Train Acc':>10} | {'Test Acc':>10} | {'Gap':>8}")
    print("-" * 47)

    for depth in depth_values:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
            ('classifier', DecisionTreeClassifier(max_depth=depth, random_state=42))
        ])

        pipeline.fit(X_train_raw, y_train)

        train_acc = accuracy_score(y_train, pipeline.predict(X_train_raw))
        test_acc = accuracy_score(y_test, pipeline.predict(X_test_raw))
        gap = train_acc - test_acc

        depth_str = str(depth) if depth is not None else "None (unlimited)"
        print(f"{depth_str:>10} | {train_acc*100:>9.2f}% | {test_acc*100:>9.2f}% | {gap*100:>7.2f}%")

if __name__ == "__main__":
    experiment_decision_tree()
