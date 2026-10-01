import os
import pandas as pd

DATA_PATH = os.path.join("data", "raw", "customer_support_tickets.csv")

def explore():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Shape (Rows, Columns): {df.shape}")
    
    print("\n--- Missing Values ---")
    print(df[['ticket_subject', 'ticket_description', 'ticket_type', 'ticket_priority']].isnull().sum())
    
    num_duplicates = df.duplicated(subset=['ticket_description']).sum()
    print(f"\n--- Duplicate Descriptions ---: {num_duplicates}")
    
    print("\n--- Class Counts (ticket_type) ---")
    print(df['ticket_type'].value_counts())
    
    print("\n--- Class Proportions (%) ---")
    print((df['ticket_type'].value_counts(normalize=True) * 100).round(2))
    
    print("\n--- Class Counts (ticket_priority) ---")
    print(df['ticket_priority'].value_counts())
    
    df['word_count'] = df['ticket_description'].str.split().apply(len)
    print("\n--- Word Count Summary Statistics ---")
    print(df['word_count'].describe())
    
    print("\n--- Sample Ticket Descriptions ---")
    for i, sample in enumerate(df['ticket_description'].head(2), start=1):
        print(f"\n[Sample {i}]:\n{sample}")

if __name__ == "__main__":
    explore()
