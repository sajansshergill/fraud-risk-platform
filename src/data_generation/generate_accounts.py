from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR
from src.utils import set_random_seed

RISK_TIERS = ["low", "medium", "high"]

def assign_risk_tier(vendor_risk_score: float, is_fraud: int) -> str:
    if is_fraud == 1 or vendor_risk_score >= 0.75:
        return "high"
    if vendor_risk_score >= 0.45:
        return "medium"
    return "low"

def assign_credit_limit(employee_count: int, declared_revenue: float, risk_tier: str) -> int:
    base_limit = min(max(int(declared_revenue * 0.02), 5_000), 250_000)
    
    if employee_count <= 3:
        base_limit = int(base_limit * 0.7)
    elif employee_count >= 50:
        base_limit = int(base_limit * 1.15)
        
    if risk_tier == "high":
        base_limit = int(base_limit * 0.45)
    elif risk_tier == "medium":
        base_limit = int(base_limit * 0.75)
        
    rounded = max(2_000, min(base_limit, 250_000))
    return int(round(rounded / 1000) * 1000)

def generate_accounts(seed: int = 42) -> pd.DataFrame:
    set_random_seed(seed)
    
    applications_path: Path = RAW_DIR / "applications.csv"
    vendor_path: Path = RAW_DIR / "vendor_risk_signals.csv"
    
    if not applications_path.exists():
        raise FileNotFoundError(
            f"{vendor_path} not found. Run generate_applications first."
        )
    if not vendor_path.exists():
        raise FileNotFoundError(
            f"{vendor_path} not found. Run generate_vendor_signals first."
        )
    apps_df = pd.read_csv(applications_path)
    vendor_df = pd.read_csv(vendor_path)
    
    df = apps_df.merge(
        vendor_df[["application_id", "vendor_risk_score"]],
        on="application_id",
        how="left",
    )
    
    rows = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        risk_tier = assign_risk_tier(
            vendor_risk_score=float(row.vendor_risk_score),
            is_fraud=int(row.is_fraud),
        )
        credit_limit = assign_credit_limit(
            employee_count=int(row.employee_count),
            declared_revenue=float(row.declared_revenue),
            risk_tier=risk_tier,
        )
        
        rows.append(
            {
                "account_id": f"ACC_{idx:06d}",
                "application_id": row.application_id,
                "business_name": row.business_name,
                "industry": row.industry,
                "approval_status": "approved",
                "assigned_credit_limit": credit_limit,
                "risk_tier": risk_tier,
                "is_fraud_account": int(row.is_fraud),
            }
        )
    return pd.DataFrame(rows)

def main() -> None:
    df = generate_accounts(seed=42)
    output_path: Path = RAW_DIR / "accounts.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Saved accounts data to: {output_path}")
    print(df.head())
    print("\nRisk tier distribution:")
    print(df["risk_tier"].value_counts(normalize=True).round(4))
    
if __name__ == "__main__":
    main()