import os
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")

def run_experiment():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train_raw = train_df['text']
    y_train = train_df['label']
    X_test_raw = test_df['text']
    y_test = test_df['label']

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    print(f"Data shapes: Train = {X_train.shape}, Test = {X_test.shape}")
    print()

    k_values = [1, 3, 5, 9, 15, 25]
    metrics = ['cosine', 'euclidean']

    print(f"{'Metric':<10} | {'k':>3} | {'Train Acc':>10} | {'Test Acc':>10} | {'Macro F1':>10} | {'Predict Time':>12}")
    print("-" * 68)

    for metric in metrics:
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
            
            knn.fit(X_train, y_train)

            start = time.perf_counter()
            test_preds = knn.predict(X_test)
            pred_time = time.perf_counter() - start

            train_preds = knn.predict(X_train)
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            macro_f1 = f1_score(y_test, test_preds, average='macro')

            print(f"{metric:<10} | {k:>3} | {train_acc*100:>9.2f}% | {test_acc*100:>9.2f}% | {macro_f1*100:>9.2f}% | {pred_time*1000:>10.2f}ms")
        print("-" * 68)

if __name__ == "__main__":
    run_experiment()
