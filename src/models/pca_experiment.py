import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
OUTPUT_PLOT = os.path.join("models", "pca_2d_clusters.png")

def run_experiment():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train_raw = train_df['text']
    y_train = train_df['label']
    X_test_raw = test_df['text']
    y_test = test_df['label']

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    total_features = X_train_tfidf.shape[1]
    print(f"Original Feature Dimension: {total_features} features\n")

    clf_full = LogisticRegression(C=10, max_iter=1000, random_state=42)
    clf_full.fit(X_train_tfidf, y_train)
    full_preds = clf_full.predict(X_test_tfidf)
    full_acc = accuracy_score(y_test, full_preds)
    full_macro = f1_score(y_test, full_preds, average='macro')

    print(f"{'Dimensions':<12} | {'Variance Kept':>14} | {'Test Acc':>10} | {'Macro F1':>10}")
    print("-" * 54)
    print(f"{'Full (' + str(total_features) + ')':<12} | {'100.0%':>14} | {full_acc*100:>9.2f}% | {full_macro*100:>9.2f}%")

    dimensions = [2, 10, 50, 100, 300, 500]

    for n in dimensions:
        svd = TruncatedSVD(n_components=n, random_state=42)
        X_train_reduced = svd.fit_transform(X_train_tfidf)
        X_test_reduced = svd.transform(X_test_tfidf)

        var_explained = svd.explained_variance_ratio_.sum() * 100

        clf = LogisticRegression(C=10, max_iter=1000, random_state=42)
        clf.fit(X_train_reduced, y_train)
        preds = clf.predict(X_test_reduced)

        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average='macro')

        print(f"{n:<12} | {var_explained:>13.2f}% | {acc*100:>9.2f}% | {macro_f1*100:>9.2f}%")

    # generate 2d visualization
    print("\nGenerating 2D visualization plot...")
    svd_2d = TruncatedSVD(n_components=2, random_state=42)
    coords_2d = svd_2d.fit_transform(X_train_tfidf)

    plt.figure(figsize=(10, 7), dpi=150)
    categories = sorted(y_train.unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for cat, color in zip(categories, colors):
        mask = (y_train == cat)
        plt.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            label=cat,
            alpha=0.6,
            edgecolors='none',
            s=35,
            c=color
        )

    plt.title("Support Ticket Clusters in 2D Space (TruncatedSVD / PCA)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel(f"Principal Component 1 ({svd_2d.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({svd_2d.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    plt.legend(title="Ticket Category", frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.savefig(OUTPUT_PLOT)
    plt.close()
    print(f"Saved 2D scatter plot to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    run_experiment()
