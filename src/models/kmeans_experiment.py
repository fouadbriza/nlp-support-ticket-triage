import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
OUTPUT_PLOT = os.path.join("models", "kmeans_elbow.png")

def run_experiment():
    train_df = pd.read_csv(TRAIN_PATH)
    texts = train_df['text']
    true_labels = train_df['label']

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    print(f"Data shape: {X.shape[0]} tickets, {X.shape[1]} features\n")

    k_range = list(range(2, 9))
    inertias = []

    print("=== Step 1: The Elbow Method (Compactness vs K) ===")
    print(f"{'K':>4} | {'Inertia':>12} | {'Change':>12}")
    print("-" * 34)

    prev_inertia = None
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

        if prev_inertia is not None:
            change = f"{(kmeans.inertia_ - prev_inertia):>10.1f}"
        else:
            change = f"{'-':>10}"

        print(f"{k:>4} | {kmeans.inertia_:>12.2f} | {change:>12}")
        prev_inertia = kmeans.inertia_

    # save elbow plot
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=7)
    plt.title("K-Means Elbow Method (Inertia vs Number of Clusters)", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Number of Clusters (K)", fontsize=11)
    plt.ylabel("Inertia (Sum of Squared Distances)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.savefig(OUTPUT_PLOT)
    plt.close()
    print(f"\nSaved elbow curve plot to: {OUTPUT_PLOT}\n")

    # fit K=5 matching our 5 real ticket classes
    print("=== Step 2: Analyzing K = 5 Clusters ===")
    k_opt = 5
    kmeans_5 = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
    cluster_labels = kmeans_5.fit_predict(X)

    order_centroids = kmeans_5.cluster_centers_.argsort()[:, ::-1]
    print("\nTop 8 Keywords Discovered per Cluster Centroid:")
    print("-" * 60)
    for i in range(k_opt):
        top_words = terms[order_centroids[i, :8]]
        print(f"Cluster {i}: {', '.join(top_words)}")

    print("\n=== Step 3: Contingency Matrix (Real Label vs Cluster) ===")
    contingency = pd.crosstab(true_labels, cluster_labels, rownames=['True Label'], colnames=['Cluster'])
    print(contingency)

    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)

    print(f"\nAdjusted Rand Index (ARI):         {ari:.4f}  (1.0 = perfect match)")
    print(f"Normalized Mutual Info (NMI):      {nmi:.4f}  (1.0 = perfect match)")

if __name__ == "__main__":
    run_experiment()
