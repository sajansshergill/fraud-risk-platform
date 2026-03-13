from __future__ import annotations

from datetime import datetime, timedelta
import random
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR
from src.utils import set_random_seed

MERCHANTS = {
    "Software": ["AWS", "OpenAI", "Notion", "Slack", "Zoom", "Figma"],
    "Office Supplies": ["Staples", "Office Depot", "Amazon Business"],
    "Travel": ["Delta", "United", "Marriott", "Uber", "Lyft"],
    "Advertising": ["Google Ads", "Meta Ads", "LinkedIn Ads"],
    "Electronics": ["Best Buy", "Apple", "Dell"],
    "Meals": ["DoorDash", "Uber Eats", "Sweetgreen"],
    "Finance Tools": ["Stripe", "Brex", "Ramp Partner", "QuickBooks"],
}

HIGH_RISK_CATEGORIES = {"Electronics", "Travel"}
COUNTRIES = ["US", "US","US", "US", 'CA', "GB", "IN"]

def random_timestamp(days_back: int = 120) -> datetime:
    now = datetime.now()
    return now - timedelta(
        days=random.randint(0, days_back),
        hours = random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    
def generate_normal_amount(category: str, credit_limit: int) -> float:
    if category == "Software":
        amount = random.uniform(20, 1200)
    elif category == "Office Supplies":
        amount = random.uniform(15, 800)
    elif category == "Travel":
        amount = random.uniform(80, 2500)
    elif category == "Advertising":
        amount = random.uniform(100, 4000)
    elif category == "Electronics":
        amount = random.uniform(150, 3500)
    elif category == "Meals":
        amount = random.uniform(20, 1500)
    elif category == "Finance Tools":
        amount = random.uniform(50, 3000)
    else:
        # Fallback for any unexpected category
        amount = random.uniform(20, 1500)

    amount = min(amount, credit_limit * random.uniform(0.02, 0.25))
    return round(max(amount, 5.0), 2)

def generate_fraud_amount(credit_limit: int) -> float:
    return round(random.uniform(0.35, 0.95) * credit_limit, 2)

def generate_transactions(seed: int = 42) -> pd.DataFrame:
    set_random_seed(seed)
    
    accounts_path: Path = RAW_DIR / "accounts.csv"
    applications_path: Path = RAW_DIR / "applications.csv"
    
    if not accounts_path.exists():
        raise FileNotFoundError(f"{accounts_path} not found. Run generate_accounts first.")
    if not applications_path.exists():
        raise FileNotFoundError(
            f"{applications_path} not found. Run generate_applications first."
        )
    
    accounts_df = pd.read_csv(accounts_path)
    apps_df = pd.read_csv(applications_path)[["application_id", "device_id", "ip_address"]]
    
    df = accounts_df.merge(apps_df, on="application_id", how="left")
    
    rows = []
    txn_counter = 1
    
    for row in df.itertuples(index=False):
        txn_count =random.randint(18, 80)
        
        for _ in range(txn_count):
            category = random.choice(list(MERCHANTS.keys()))
            merchant = random.choice(MERCHANTS[category])
            
            is_fraud = 0
            country = random.choice(COUNTRIES)
            device_id = row.device_id
            ip_address = row.ip_address
            
            fraud_profile = (
                int(row.is_fraud_account) == 1 and random.random() < 0.18
            ) or random.random() < 0.015
            
            if fraud_profile:
                is_fraud = 1
                category = random.choice(list(HIGH_RISK_CATEGORIES))
                merchant = random.choice(MERCHANTS[category])
                amount = generate_fraud_amount(int(row.assigned_credit_limit))
                
                if random.random() < 0.65:
                    country = random.choice(["GB", "IN", "BR", "NG", "SG"])
                    
                if random.random() < 0.60:
                    ip_address = ".".join(str(random.randint(1, 255)) for _ in range(4))
            else:
                amount = generate_normal_amount(
                    category = category,
                    credit_limit=int(row.assigned_credit_limit),
                )
            
            rows.append(
                {
                    "transaction_id": f"TXN_{txn_counter:07d}",
                    "account_id": row.account_id,
                    "application_id": row.application_id,
                    "merchant_name": merchant,
                    "merchant_category": category,
                    "amount": amount,
                    "country": country,
                    "currency": "USD",
                    "device_id": device_id,
                    "ip_address": ip_address,
                    "transaction_timestamp": random_timestamp().isoformat(),
                    "assigned_credit_limit": int(row.assigned_credit_limit),
                    "account_risk_tier": row.risk_tier,
                    "is_fraud_transaction": is_fraud,
                    
                }
            )
            txn_counter += 1

    txns_df = pd.DataFrame(rows)
    txns_df = txns_df.sort_values("transaction_timestamp").reset_index(drop=True)
    return txns_df

def main() -> None:
    df = generate_transactions(seed=42)
    output_path = RAW_DIR / "transactions.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Saved transactions data to: {output_path}")
    print(df.head())
    print("\nFraud rate:")
    print(df["is_fraud_transaction"].value_counts(normalize=True).round(4))
    

if __name__ == "__main__":
    main()
        