from __future__ import annotations

import random
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from src.config import RAW_DIR
from src.utils import set_random_seed

PUBLIC_EMAIL_DOMAINS = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com"}

def extract_email_domain(email: str) -> str:
    return email.split("@")[-1].strip().lower()

def normalize_domain(domain: str) -> str:
    parsed = urlparse(domain if domain.startswith("http") else f"https://{domain}")
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host

def clip_score(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 4)

def build_vendor_row(app_row: pd.Series, docs_lookup: dict[str, dict]) -> dict:
    is_fraud = int(app_row["is_fraud"]) == 1
    email = app_row["email"]
    normalized_domain = normalize_domain(app_row["domain"])
    email_domain = extract_email_domain(email)
    
    employee_count = int(app_row["employee_count"])
    declared_revenue = float(app_row["declared_revenue"])
    
    doc_row = docs_lookup.get(app_row["application_id"], {})
    doc_name_mismatch_flag = int(doc_row.get("doc_name_mismatch_flag", 0))
    doc_address_mismatch_flag = int(doc_row.get("doc_address_mismatch_flag", 0))
    metadata_anomaly_score = float(doc_row.get("metadata_anomaly_score", 0.1))
    
    # base signals
    if is_fraud:
        domain_age_days = random.randint(1, 120)
        ip_risk_score = random.uniform(0.55, 0.98)
        device_risk_score = random.uniform(0.50, 0.97)
        email_reputation_score = random.uniform(0.10, 0.55)
        business_registry_match_score = random.uniform(0.20, 0.75)
    else:
        domain_age_days = random.randint(180, 5000)
        ip_risk_score = random.uniform(0.55, 0.98)
        device_risk_score = random.uniform(0.50, 0.97)
        email_reputation_score = random.uniform(0.10, 0.55)
        business_registry_match_score = random.uniform(0.20, 0.75)
        
    if email_domain in PUBLIC_EMAIL_DOMAINS:
        email_reputation_score -= random.uniform(0.10, 0.30)
        business_registry_match_score -= random.uniform(0.05, 0.15)
        
    if email_domain != normalized_domain:
        business_registry_match_score -= random.uniform(0.05, 0.15)
    
    if employee_count <= 3 and declared_revenue > 5_000_000:
        ip_risk_score += random.uniform(0.05, 0.15)
        device_risk_score += random.uniform(0.05, 0.15)
    
    if doc_name_mismatch_flag == 1:
        business_registry_match_score -= random.uniform(0.10, 0.30)
        device_risk_score += random.uniform(0.03, 0.12)
        
    if doc_address_mismatch_flag == 1:
        ip_risk_score += random.uniform(0.03, 0.10)
        
    vendor_risk_score = (
        0.22 * (1 - min(domain_age_days / 3650, 1.0))
        + 0.18 * (1 - clip_score(email_reputation_score))
        + 0.18 * clip_score(ip_risk_score)
        + 0.18 * clip_score(device_risk_score)
        + 0.14 * (1 - clip_score(business_registry_match_score))
        + 0.10 * clip_score(metadata_anomaly_score)
    )
    
    return {
        "application_id": app_row["application_id"],
        "domain_age_days": domain_age_days,
        "email_domain": email_domain,
        "domain_email_match_flag": int(email_domain == normalized_domain),
        "email_reputation_score": clip_score(email_reputation_score),
        "ip_risk_score": clip_score(ip_risk_score),
        "device_risk_score": clip_score(device_risk_score),
        "business_registry_match_score": clip_score(business_registry_match_score),
        "vendor_risk_score": clip_score(vendor_risk_score),
    }
    
def generate_vendor_signals(seed: int = 42) -> pd.DataFrame:
    set_random_seed(seed)
    
    applications_path: Path = RAW_DIR / "applications.csv"
    documents_path: Path = RAW_DIR / "documents.csv"
    
    if not applications_path.exists():
        raise FileNotFoundError(
            f"{applications_path} not found. Run generate_applications first."
        )
    if not documents_path.exists():
        raise FileNotFoundError(
            f"{documents_path} not found. Run generate_documents first."
        )
    
    apps_df = pd.read_csv(applications_path)
    docs_df = pd.read_csv(documents_path)
    
    docs_lookup = docs_df.set_index("application_id").to_dict(orient="index")
    rows = [build_vendor_row(row, docs_lookup) for _, row in apps_df.iterrows()]
    
    return pd.DataFrame(rows)

def main() -> None:
    df = generate_vendor_signals(seed=42)
    output_path: Path = RAW_DIR / "vendor_risk_signals.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Saved vendor signals data to: {output_path}")
    print(df.head())
    print("\n Average vendor risk score:", round(df["vendor_risk_score"].mean(), 4))
    
if __name__ == "__main__":
    main()