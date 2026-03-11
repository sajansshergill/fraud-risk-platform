from __future__ import annotations

import random
from pathlib import Path
from typing import List

import pandas as pd
from faker import Faker
from rapidfuzz.fuzz import ratio

from src.config import RAW_DIR
from src.utils import set_random_seed

fake = Faker()

DOCUMENT_TYPES: List[str] = [
    "EIN Letter",
    "Certificate of Incorporation",
    "Bank Statement",
]

STREET_SUFFIX_REPLACEMENTS = {
    "Street": "St",
    "Avenue": "Ave",
    "Road": "Rd",
    "Boulevard": "Blvd",
    "Drive": "Dr",
    "Lane": "Ln",
}

def corrupt_address(address: str) -> str:
    updated = address
    for full, short in STREET_SUFFIX_REPLACEMENTS.items():
        updated = updated.replace(full, short)
        
    if random.random() < 0.35:
        updated = updated.replace(",", "")
        
    if random.random() < 0.30:
        updated = f"{random.randint(10, 9999)} {updated}"
    
    return updated


def corrupt_business_name(name: str) -> str:
    updated = name
    # randomly drop a word or add a suffix/prefix to simulate OCR / fraud issues
    parts = updated.split()
    if len(parts) > 1 and random.random() < 0.5:
        updated = " ".join(parts[:-1])
    elif random.random() < 0.5:
        updated = f"{updated} Holdings"
    # introduce a small character-level corruption
    if len(updated) > 3 and random.random() < 0.5:
        idx = random.randint(0, len(updated) - 2)
        updated = updated[:idx] + updated[idx + 1] + updated[idx] + updated[idx + 2 :]
    return updated

def build_document_row(app_row: pd.Series) -> dict:
    risky = int(app_row["is_fraud"]) == 1
    
    business_name = app_row["business_name"]
    business_address = fake.address().replace("\n", ", ")
    
    ocr_business_name = business_name
    ocr_address = business_address
    
    metadata_anomaly_score = round(random.uniform(0.0, 0.25), 4)
    validation_score = round(random.uniform(0.78, 0.99), 4)
    ocr_confidence_score = round(random.uniform(0.85, 0.99), 4)
    
    if risky and random.random() < 0.65:
        ocr_business_name = corrupt_business_name(business_name)
        metadata_anomaly_score = round(random.uniform(0.45, 0.95), 4)
        validation_score = round(random.uniform(0.25, 0.75), 4)
        ocr_confidence_score = round(random.uniform(0.60, 0.92), 4)
        
    if risky and random.random() < 0.50:
        ocr_address = corrupt_address(business_address)
        metadata_anomaly_score = max(
            metadata_anomaly_score, round(random.uniform(0.40, 0.98), 4)
        )
    
    business_name_similarity = round(ratio(business_name, ocr_business_name) / 100, 4)
    address_similarity = round(ratio(business_address, ocr_address) / 100, 4)
    
    return {
        "doc_id": f"DOC_{fake.uuid4()[:12]}",
        "application_id": app_row["application_id"],
        "document_type": random.choice(DOCUMENT_TYPES),
        "document_business_name": business_name,
        "document_address": business_address,
        "ocr_business_name": ocr_business_name,
        "ocr_address": ocr_address,
        "business_name_similarity": business_name_similarity,
        "address_similarity": address_similarity,
        "metadata_anomaly_score": metadata_anomaly_score,
        "ocr_confidence_score": ocr_confidence_score,
        "validation_score": validation_score,
        "doc_name_mismatch_flag": int(business_name_similarity < 0.88),
        "doc_address_mismatch_flag": int(address_similarity < 0.80),
    }
    
def generate_documents(seed: int = 42) -> pd.DataFrame:
    set_random_seed(seed)
    Faker.seed(seed)
    
    applications_path: Path = RAW_DIR / "applications.csv"
    if not applications_path.exists():
        raise FileNotFoundError(
            f"{applications_path} not found. Run generate_applications first."
        )
        
    apps_df = pd.read_csv(applications_path)
    docs = [build_document_row(row) for _, row in apps_df.iterrows()]
    return pd.DataFrame(docs)

def main() -> None:
    df = generate_documents(seed=42)
    output_path: Path = RAW_DIR / "documents.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Saved documents data to: {output_path}")
    print(df.head())
    print("\n Mismatch rates:")
    print(df[["doc_name_mismatch_flag", "doc_address_mismatch_flag"]].mean().round(4))
    
if __name__ == "__main__":
    main()