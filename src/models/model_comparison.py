import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
MODELS_DIR = "models"

def run_comparison():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df['text']
    y_train = train_df['label']
    X_test = test_df['text']
    y_test = test_df['label']

    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples.")
    print("Classes:", sorted(y_test.unique()))
    print()

    results = []
    class_reports = {}

    dummy = DummyClassifier(strategy="most_frequent")
    t0 = time.perf_counter()
    dummy.fit(X_train, y_train)
    t1 = time.perf_counter()
    y_pred_dummy = dummy.predict(X_test)
    dummy_infer_time = (time.perf_counter() - t1) * 1000 / len(X_test)

    results.append({
        "Model": "Baseline (Most Frequent)",
        "Accuracy": accuracy_score(y_test, y_pred_dummy),
        "Precision (Macro)": precision_score(y_test, y_pred_dummy, average='macro', zero_division=0),
        "Recall (Macro)": recall_score(y_test, y_pred_dummy, average='macro', zero_division=0),
        "F1 (Macro)": f1_score(y_test, y_pred_dummy, average='macro', zero_division=0),
        "F1 (Weighted)": f1_score(y_test, y_pred_dummy, average='weighted', zero_division=0),
        "Inference (ms/sample)": dummy_infer_time,
        "Model Size (KB)": 0.0
    })
    class_reports["Baseline"] = classification_report(y_test, y_pred_dummy, zero_division=0, output_dict=True)

    models_to_test = [
        ("Logistic Regression (C=10)", "logistic_regression_pipeline.joblib", False),
        ("Decision Tree (max_depth=50)", "decision_tree_pipeline.joblib", False),
        ("Random Forest (100 trees)", "random_forest_pipeline.joblib", False),
        ("XGBoost (100 trees)", "xgboost_pipeline.joblib", True),
    ]

    for model_name, filename, is_xgb in models_to_test:
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            continue

        file_size_kb = os.path.getsize(path) / 1024

        loaded = joblib.load(path)
        if is_xgb:
            pipeline = loaded['pipeline']
            le = loaded['label_encoder']
            y_test_encoded = le.transform(y_test)

            t0 = time.perf_counter()
            y_pred_encoded = pipeline.predict(X_test)
            infer_ms_per_sample = (time.perf_counter() - t0) * 1000 / len(X_test)

            y_pred = le.inverse_transform(y_pred_encoded)
        else:
            pipeline = loaded
            t0 = time.perf_counter()
            y_pred = pipeline.predict(X_test)
            infer_ms_per_sample = (time.perf_counter() - t0) * 1000 / len(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            "Model": model_name,
            "Accuracy": acc,
            "Precision (Macro)": prec,
            "Recall (Macro)": rec,
            "F1 (Macro)": f1_m,
            "F1 (Weighted)": f1_w,
            "Inference (ms/sample)": infer_ms_per_sample,
            "Model Size (KB)": file_size_kb
        })
        class_reports[model_name] = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    results_df = pd.DataFrame(results)

    print("=================================== FINAL BENCHMARK COMPARISON TABLE ===================================")
    print(f"{'Model':<30} | {'Accuracy':>9} | {'Precision':>9} | {'Recall':>9} | {'F1 (Macro)':>10} | {'F1 (Wtd)':>9} | {'Lat(ms)':>8} | {'Size':>9}")
    print("-" * 105)
    for _, row in results_df.iterrows():
        size_str = f"{row['Model Size (KB)']/1024:.2f} MB" if row['Model Size (KB)'] >= 1024 else f"{row['Model Size (KB)']:.1f} KB"
        print(f"{row['Model']:<30} | {row['Accuracy']*100:>8.2f}% | {row['Precision (Macro)']*100:>8.2f}% | {row['Recall (Macro)']*100:>8.2f}% | {row['F1 (Macro)']*100:>9.2f}% | {row['F1 (Weighted)']*100:>8.2f}% | {row['Inference (ms/sample)']:>7.3f} | {size_str:>9}")

    print("\n=================================== PER-CLASS F1 SCORE BREAKDOWN ===================================")
    classes = sorted(y_test.unique())
    header = f"{'Model':<30} | " + " | ".join([f"{c:>15}" for c in classes])
    print(header)
    print("-" * len(header))
    for model_name, rep in class_reports.items():
        row_str = f"{model_name:<30} | "
        scores = []
        for c in classes:
            f1 = rep.get(c, {}).get('f1-score', 0.0) * 100
            scores.append(f"{f1:>14.2f}%")
        row_str += " | ".join(scores)
        print(row_str)

if __name__ == "__main__":
    run_comparison()
