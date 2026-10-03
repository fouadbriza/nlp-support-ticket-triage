import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")

def run_experiment():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df['text']
    y_train = train_df['label']
    X_test = test_df['text']
    y_test = test_df['label']

    print("=== Training Class Distribution ===")
    counts = y_train.value_counts()
    for label, count in counts.items():
        print(f"  {label:<18}: {count:>4} ({count/len(y_train)*100:.1f}%)")
    print(f"  {'Total':<18}: {len(y_train):>4}")
    print()

    models = {
        "Logistic Regression (default)": LogisticRegression(C=10, max_iter=1000, random_state=42),
        "Logistic Regression (balanced)": LogisticRegression(C=10, class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest (default)": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Random Forest (balanced)": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    }

    print(f"{'Model':<32} | {'Accuracy':>10} | {'Macro F1':>10} | {'Weighted F1':>12}")
    print("-" * 72)

    predictions = {}
    for name, clf in models.items():
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
            ('classifier', clf)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        predictions[name] = preds

        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average='macro')
        weighted_f1 = f1_score(y_test, preds, average='weighted')

        print(f"{name:<32} | {acc*100:>9.2f}% | {macro_f1*100:>9.2f}% | {weighted_f1*100:>11.2f}%")

    print("\n" + "=" * 72)
    print("Per-Class Recall Comparison (Default vs Balanced)")
    print("=" * 72)
    labels = sorted(y_test.unique())
    print(f"{'Class':<18} | {'LR Default':>12} | {'LR Balanced':>12} | {'RF Default':>12} | {'RF Balanced':>12}")
    print("-" * 72)

    from sklearn.metrics import recall_score
    recalls = {}
    for name in models:
        recalls[name] = recall_score(y_test, predictions[name], average=None, labels=labels)

    for i, label in enumerate(labels):
        print(f"{label:<18} | "
              f"{recalls['Logistic Regression (default)'][i]*100:>11.2f}% | "
              f"{recalls['Logistic Regression (balanced)'][i]*100:>11.2f}% | "
              f"{recalls['Random Forest (default)'][i]*100:>11.2f}% | "
              f"{recalls['Random Forest (balanced)'][i]*100:>11.2f}%")

if __name__ == "__main__":
    run_experiment()
