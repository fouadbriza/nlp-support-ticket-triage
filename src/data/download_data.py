import os
import json
import urllib.request
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
DESTINATION_PATH = os.path.join("data", "raw", "customer_support_tickets.csv")

INTENT_TO_CATEGORY = {
    "pay_bill": "billing",
    "bill_balance": "billing",
    "bill_due": "billing",
    "min_payment": "billing",
    "account_blocked": "account",
    "freeze_account": "account",
    "transfer": "account",
    "cancel": "cancellation",
    "cancel_reservation": "cancellation",
    "report_lost_card": "refund",
    "damaged_card": "refund",
    "card_declined": "refund",
    "credit_score": "product_inquiry",
    "improve_credit_score": "product_inquiry",
    "credit_limit": "product_inquiry",
    "credit_limit_change": "product_inquiry",
    "new_card": "product_inquiry",
    "replacement_card_duration": "product_inquiry",
}

def download_dataset():
    print("Downloading CLINC150 dataset...")
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read())

    all_items = data["train"] + data["val"] + data["test"]

    rows = []
    for text, intent in all_items:
        if intent in INTENT_TO_CATEGORY:
            rows.append({"text": text, "ticket_type": INTENT_TO_CATEGORY[intent]})

    df = pd.DataFrame(rows)
    print(f"Total filtered examples: {len(df)}")
    print(f"Category distribution:\n{df['ticket_type'].value_counts()}")

    os.makedirs(os.path.dirname(DESTINATION_PATH), exist_ok=True)
    df.to_csv(DESTINATION_PATH, index=False)
    print(f"Saved to: {DESTINATION_PATH}")

if __name__ == "__main__":
    download_dataset()
