from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR


HIGH_RISK_COUNTRIES = {"GB", "IN", "BR", "NG", "SG"}
HIGH_RISK_MERCHANT_CATEGORIES = {"Electronics", "Travel"}


def load_raw_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transactions_path: Path = RAW_DIR / "transactions.csv"
    accounts_path: Path = RAW_DIR / "accounts.csv"
    vendor_path: Path = RAW_DIR / "vendor_risk_signals.csv"

    if not transactions_path.exists():
        raise FileNotFoundError(
            f"{transactions_path} not found. Run generate_transactions first."
        )
    if not accounts_path.exists():
        raise FileNotFoundError(
            f"{accounts_path} not found. Run generate_accounts first."
        )
    if not vendor_path.exists():
        raise FileNotFoundError(
            f"{vendor_path} not found. Run generate_vendor_signals first."
        )

    txns_df = pd.read_csv(transactions_path)
    accounts_df = pd.read_csv(accounts_path)
    vendor_df = pd.read_csv(vendor_path)

    return txns_df, accounts_df, vendor_df


def build_payments_features() -> pd.DataFrame:
    txns_df, accounts_df, vendor_df = load_raw_tables()

    df = (
        txns_df.merge(
            accounts_df[
                [
                    "account_id",
                    "application_id",
                    "industry",
                    "risk_tier",
                    "is_fraud_account",
                ]
            ],
            on=["account_id", "application_id"],
            how="left",
        )
        .merge(
            vendor_df[
                [
                    "application_id",
                    "vendor_risk_score",
                    "ip_risk_score",
                    "device_risk_score",
                    "email_reputation_score",
                ]
            ],
            on="application_id",
            how="left",
        )
        .copy()
    )

    df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
    df = df.sort_values(["account_id", "transaction_timestamp"]).reset_index(drop=True)

    df["country_risk_flag"] = df["country"].isin(HIGH_RISK_COUNTRIES).astype(int)
    df["high_risk_merchant_flag"] = (
        df["merchant_category"].isin(HIGH_RISK_MERCHANT_CATEGORIES).astype(int)
    )
    df["amount_to_limit_ratio"] = (
        df["amount"] / df["assigned_credit_limit"].replace(0, np.nan)
    ).fillna(0.0)

    account_avg = df.groupby("account_id")["amount"].transform("mean")
    account_std = df.groupby("account_id")["amount"].transform("std").fillna(1.0)
    df["amount_zscore_vs_account"] = ((df["amount"] - account_avg) / account_std).replace([np.inf, -np.inf], 0).fillna(0)

    df["txn_count_1d"] = df.groupby("account_id").cumcount() + 1

    account_country_nunique = df.groupby("account_id")["country"].transform("nunique")
    df["country_diversity_score"] = account_country_nunique

    device_counts = df.groupby("account_id")["device_id"].transform("nunique")
    df["device_diversity_score"] = device_counts

    merchant_counts = df.groupby("account_id")["merchant_name"].transform("nunique")
    df["merchant_diversity_score"] = merchant_counts

    df["new_device_flag"] = (
        df.groupby("account_id")["device_id"].transform("first") != df["device_id"]
    ).astype(int)

    df["new_ip_flag"] = (
        df.groupby("account_id")["ip_address"].transform("first") != df["ip_address"]
    ).astype(int)

    df["high_vendor_risk_flag"] = (df["vendor_risk_score"] >= 0.65).astype(int)
    df["high_ip_risk_flag"] = (df["ip_risk_score"] >= 0.70).astype(int)
    df["high_device_risk_flag"] = (df["device_risk_score"] >= 0.70).astype(int)
    df["low_email_reputation_flag"] = (df["email_reputation_score"] < 0.45).astype(int)

    df["behavior_risk_score"] = (
        0.25 * df["country_risk_flag"]
        + 0.20 * df["new_device_flag"]
        + 0.20 * df["new_ip_flag"]
        + 0.20 * np.clip(df["amount_to_limit_ratio"], 0, 1)
        + 0.15 * df["high_risk_merchant_flag"]
    ).round(4)

    df["payment_risk_score"] = (
        0.25 * np.clip(df["amount_to_limit_ratio"], 0, 1)
        + 0.20 * np.clip((df["amount_zscore_vs_account"] / 5), 0, 1)
        + 0.15 * df["country_risk_flag"]
        + 0.15 * df["new_device_flag"]
        + 0.10 * df["new_ip_flag"]
        + 0.15 * df["high_risk_merchant_flag"]
    ).round(4)

    feature_cols = [
        "transaction_id",
        "account_id",
        "application_id",
        "industry",
        "risk_tier",
        "merchant_category",
        "country",
        "amount",
        "assigned_credit_limit",
        "amount_to_limit_ratio",
        "amount_zscore_vs_account",
        "txn_count_1d",
        "country_diversity_score",
        "device_diversity_score",
        "merchant_diversity_score",
        "country_risk_flag",
        "high_risk_merchant_flag",
        "new_device_flag",
        "new_ip_flag",
        "vendor_risk_score",
        "ip_risk_score",
        "device_risk_score",
        "email_reputation_score",
        "high_vendor_risk_flag",
        "high_ip_risk_flag",
        "high_device_risk_flag",
        "low_email_reputation_flag",
        "behavior_risk_score",
        "payment_risk_score",
        "is_fraud_account",
        "is_fraud_transaction",
    ]

    features_df = df[feature_cols].copy()

    numeric_cols = [col for col in feature_cols if col not in {
        "transaction_id", "account_id", "application_id", "industry", "risk_tier", "merchant_category", "country"
    }]

    for col in numeric_cols:
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

    features_df = features_df.fillna(
        {
            "industry": "Unknown",
            "risk_tier": "unknown",
            "merchant_category": "Unknown",
            "country": "Unknown",
        }
    )

    fill_numeric = [col for col in features_df.columns if col not in {
        "transaction_id", "account_id", "application_id", "industry", "risk_tier", "merchant_category", "country"
    }]
    features_df[fill_numeric] = features_df[fill_numeric].fillna(0)

    return features_df


def main() -> None:
    df = build_payments_features()
    output_path = PROCESSED_DIR / "payments_features.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved payments features to: {output_path}")
    print(df.head())
    print("\nFraud rate:")
    print(df["is_fraud_transaction"].value_counts(normalize=True).round(4))


if __name__ == "__main__":
    main()