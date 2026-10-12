import os
import joblib
import pandas as pd
import numpy as np

TEST_PATH = os.path.join("data", "processed", "test.csv")
MODEL_PATH = os.path.join("models", "logistic_regression_pipeline.joblib")

def run_error_analysis():
    test_df = pd.read_csv(TEST_PATH)
    pipeline = joblib.load(MODEL_PATH)

    X_test = test_df['text']
    y_true = test_df['label']

    classes = pipeline.classes_
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Identify errors
    errors_mask = (y_true != y_pred)
    total_samples = len(test_df)
    total_errors = errors_mask.sum()

    print("=" * 80)
    print(f"ERROR ANALYSIS FOR LOGISTIC REGRESSION (BEST MODEL)")
    print(f"Total Test Samples: {total_samples}")
    print(f"Total Misclassifications: {total_errors} ({total_errors / total_samples * 100:.2f}% error rate)")
    print("=" * 80)

    if total_errors == 0:
        print("No errors found on the test set!")
        return

    error_indices = np.where(errors_mask)[0]
    
    print("\nDETAILED BREAKDOWN OF EVERY MISCLASSIFIED SAMPLE:\n")
    for i, idx in enumerate(error_indices, 1):
        text = X_test.iloc[idx]
        actual = y_true.iloc[idx]
        predicted = y_pred[idx]
        probas = y_proba[idx]

        # Top 3 predicted classes by probability
        top_sorted_indices = np.argsort(probas)[::-1]
        
        pred_conf = probas[classes == predicted][0]
        actual_conf = probas[classes == actual][0]

        print(f"--- [Error #{i}] Sample Index: {idx} ---")
        print(f"Ticket Text : \"{text}\"")
        print(f"Actual Label: {actual} (Model assigned: {actual_conf*100:.2f}%)")
        print(f"Predicted   : {predicted} (Model assigned: {pred_conf*100:.2f}%)")
        print("Top Probabilities:")
        for rank in range(min(3, len(classes))):
            c_idx = top_sorted_indices[rank]
            print(f"   {rank+1}. {classes[c_idx]:<16}: {probas[c_idx]*100:.2f}%")
        print()

    # Confusion breakdown among errors
    error_pairs = pd.Series(
        [f"{act}  -->  {pred}" for act, pred in zip(y_true[errors_mask], y_pred[errors_mask])]
    ).value_counts()

    print("=" * 80)
    print("ERROR PATTERNS (Actual --> Predicted):")
    print("-" * 80)
    for pair, count in error_pairs.items():
        print(f"  {pair:<35} : {count} occurrences")
    print("=" * 80)

if __name__ == "__main__":
    run_error_analysis()
