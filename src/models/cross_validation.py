import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

TRAIN_PATH = os.path.join("data", "processed", "train.csv")

def compare_models():
    train_df = pd.read_csv(TRAIN_PATH)
    X = train_df['text']
    y = LabelEncoder().fit_transform(train_df['label']) # for XGBoost (doesn't support string labels)

    models = {
        "Logistic Regression (C=10)":   LogisticRegression(C=10.0, max_iter=1000, random_state=42),
        "Decision Tree (depth=10)":     DecisionTreeClassifier(max_depth=10, random_state=42),
        "Decision Tree (depth=50)":     DecisionTreeClassifier(max_depth=50, random_state=42),
        "Decision Tree (unlimited)":    DecisionTreeClassifier(max_depth=None, random_state=42),
        "Random Forest (100 trees)":     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Random Forest (300 trees)":     RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "XGBoost (100 trees, lr=0.1)":   XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='mlogloss', verbosity=0),
        "XGBoost (300 trees, lr=0.1)":   XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='mlogloss', verbosity=0),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"{'Model':<35} | {'Mean Acc':>10} | {'Std Dev':>10}")
    print("-" * 62)

    for name, clf in models.items():
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
            ('clf', clf)
        ])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"{name:<35} | {scores.mean()*100:>9.2f}% | +/- {scores.std()*100:.2f}%")

if __name__ == "__main__":
    compare_models()
