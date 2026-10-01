import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = os.path.join("data", "raw", "customer_support_tickets.csv")
PROCESSED_DIR = os.path.join("data", "processed")

def prepare_data():
    df = pd.read_csv(RAW_DATA_PATH)

    df_clean = df.drop_duplicates(subset=['text']).copy()
    print(f"Original rows: {len(df)} | After deduplication: {len(df_clean)}")

    X = df_clean['text']
    y = df_clean['ticket_type']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)} rows")
    print(f"Test set size: {len(X_test)} rows")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test})

    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    print("Saved clean train.csv and test.csv to data/processed/")

if __name__ == "__main__":
    prepare_data()
