import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

TRAIN_PATH = os.path.join("data", "processed", "train.csv")

def evaluate_cross_validation():
    train_df = pd.read_csv(TRAIN_PATH)
    X = train_df['text']
    y = train_df['label']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('classifier', LogisticRegression(C=10.0, max_iter=1000, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Running 5-Fold Stratified Cross-Validation on Logistic Regression...")
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    print("\nResults across 5 folds:")
    for fold_num, score in enumerate(scores, 1):
        print(f"  Fold {fold_num}: {score * 100:.2f}%")

    print("-" * 30)
    print(f"Mean Accuracy: {scores.mean() * 100:.2f}%")
    print(f"Standard Deviation: (+/- {scores.std() * 100:.2f}%)")

if __name__ == "__main__":
    evaluate_cross_validation()
