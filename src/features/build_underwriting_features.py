from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR


PUBLIC_EMAIL_DOMAINS = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com"}


def extract_email_domain(email: str) -> str:
    return str(email).split("@")[-1].strip().lower()


def compute_business_age_days(series: pd.Series) -> pd.Series:
    incorporation_dates = pd.to_datetime(series, errors="coerce")
    today = pd.Timestamp.today().normalize()
    return (today - incorporation_dates).dt.days.fillna(-1).astype(int)


def load_raw_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    applications_path: Path = RAW_DIR / "applications.csv"
    documents_path: Path = RAW_DIR / "documents.csv"
    vendor_path: Path = RAW_DIR / "vendor_risk_signals.csv"

    if not applications_path.exists():
        raise FileNotFoundError(
            f"{applications_path} not found. Run generate_applications first."
        )
    if not documents_path.exists():
        raise FileNotFoundError(
            f"{documents_path} not found. Run generate_documents first."
        )
    if not vendor_path.exists():
        raise FileNotFoundError(
            f"{vendor_path} not found. Run generate_vendor_signals first."
        )

    apps_df = pd.read_csv(applications_path)
    docs_df = pd.read_csv(documents_path)
    vendor_df = pd.read_csv(vendor_path)

    return apps_df, docs_df, vendor_df


def build_underwriting_features() -> pd.DataFrame:
    apps_df, docs_df, vendor_df = load_raw_tables()

    df = (
        apps_df.merge(docs_df, on="application_id", how="left")
        .merge(vendor_df, on="application_id", how="left")
        .copy()
    )

    df["email_domain_extracted"] = df["email"].apply(extract_email_domain)
    df["business_age_days"] = compute_business_age_days(df["incorporation_date"])

    df["is_public_email_domain"] = (
        df["email_domain_extracted"].isin(PUBLIC_EMAIL_DOMAINS).astype(int)
    )

    df["revenue_per_employee"] = np.where(
        df["employee_count"] > 0,
        df["declared_revenue"] / df["employee_count"],
        df["declared_revenue"],
    )

    df["high_revenue_low_headcount_flag"] = (
        (df["employee_count"] <= 3) & (df["declared_revenue"] > 5_000_000)
    ).astype(int)

    df["new_business_flag"] = (df["business_age_days"] < 90).astype(int)
    df["very_new_business_flag"] = (df["business_age_days"] < 30).astype(int)

    df["low_email_reputation_flag"] = (df["email_reputation_score"] < 0.45).astype(int)
    df["high_ip_risk_flag"] = (df["ip_risk_score"] > 0.70).astype(int)
    df["high_device_risk_flag"] = (df["device_risk_score"] > 0.70).astype(int)
    df["low_registry_match_flag"] = (
        df["business_registry_match_score"] < 0.60
    ).astype(int)
    df["high_vendor_risk_flag"] = (df["vendor_risk_score"] > 0.65).astype(int)

    ip_counts = df["ip_address"].value_counts(dropna=False)
    device_counts = df["device_id"].value_counts(dropna=False)

    df["applications_per_ip"] = df["ip_address"].map(ip_counts).fillna(1).astype(int)
    df["applications_per_device"] = (
        df["device_id"].map(device_counts).fillna(1).astype(int)
    )

    df["shared_ip_flag"] = (df["applications_per_ip"] >= 3).astype(int)
    df["shared_device_flag"] = (df["applications_per_device"] >= 3).astype(int)

    df["doc_risk_score"] = (
        0.35 * df["doc_name_mismatch_flag"].fillna(0)
        + 0.20 * df["doc_address_mismatch_flag"].fillna(0)
        + 0.25 * df["metadata_anomaly_score"].fillna(0.0)
        + 0.20 * (1 - df["validation_score"].fillna(1.0))
    ).round(4)

    df["identity_risk_score"] = (
        0.25 * df["is_public_email_domain"]
        + 0.20 * df["new_business_flag"]
        + 0.15 * df["high_revenue_low_headcount_flag"]
        + 0.20 * (1 - df["domain_email_match_flag"].fillna(1))
        + 0.20 * (1 - df["business_registry_match_score"].fillna(1.0))
    ).round(4)

    df["behavior_risk_score"] = (
        0.50 * np.minimum(df["applications_per_ip"], 5) / 5
        + 0.50 * np.minimum(df["applications_per_device"], 5) / 5
    ).round(4)

    feature_columns = [
        "application_id",
        "industry",
        "employee_count",
        "declared_revenue",
        "business_age_days",
        "revenue_per_employee",
        "is_public_email_domain",
        "high_revenue_low_headcount_flag",
        "new_business_flag",
        "very_new_business_flag",
        "domain_age_days",
        "domain_email_match_flag",
        "email_reputation_score",
        "ip_risk_score",
        "device_risk_score",
        "business_registry_match_score",
        "vendor_risk_score",
        "doc_name_mismatch_flag",
        "doc_address_mismatch_flag",
        "business_name_similarity",
        "address_similarity",
        "metadata_anomaly_score",
        "ocr_confidence_score",
        "validation_score",
        "applications_per_ip",
        "applications_per_device",
        "shared_ip_flag",
        "shared_device_flag",
        "low_email_reputation_flag",
        "high_ip_risk_flag",
        "high_device_risk_flag",
        "low_registry_match_flag",
        "high_vendor_risk_flag",
        "doc_risk_score",
        "identity_risk_score",
        "behavior_risk_score",
        "is_fraud",
    ]

    features_df = df[feature_columns].copy()

    numeric_columns = [
        "employee_count",
        "declared_revenue",
        "business_age_days",
        "revenue_per_employee",
        "domain_age_days",
        "domain_email_match_flag",
        "email_reputation_score",
        "ip_risk_score",
        "device_risk_score",
        "business_registry_match_score",
        "vendor_risk_score",
        "doc_name_mismatch_flag",
        "doc_address_mismatch_flag",
        "business_name_similarity",
        "address_similarity",
        "metadata_anomaly_score",
        "ocr_confidence_score",
        "validation_score",
        "applications_per_ip",
        "applications_per_device",
        "shared_ip_flag",
        "shared_device_flag",
        "low_email_reputation_flag",
        "high_ip_risk_flag",
        "high_device_risk_flag",
        "low_registry_match_flag",
        "high_vendor_risk_flag",
        "doc_risk_score",
        "identity_risk_score",
        "behavior_risk_score",
        "is_public_email_domain",
        "high_revenue_low_headcount_flag",
        "new_business_flag",
        "very_new_business_flag",
        "is_fraud",
    ]

    for col in numeric_columns:
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

    features_df = features_df.fillna(
        {
            "industry": "Unknown",
            "business_name_similarity": 0.0,
            "address_similarity": 0.0,
            "metadata_anomaly_score": 0.0,
            "ocr_confidence_score": 0.0,
            "validation_score": 0.0,
        }
    )

    numeric_fill_cols = [col for col in features_df.columns if col != "industry" and col != "application_id"]
    features_df[numeric_fill_cols] = features_df[numeric_fill_cols].fillna(0)

    return features_df


def main() -> None:
    features_df = build_underwriting_features()
    output_path = PROCESSED_DIR / "underwriting_features.csv"
    features_df.to_csv(output_path, index=False)

    print(f"Saved underwriting features to: {output_path}")
    print(features_df.head())
    print("\nShape:", features_df.shape)
    print("\nFraud rate:")
    print(features_df["is_fraud"].value_counts(normalize=True).round(4))


if __name__ == "__main__":
    main()
