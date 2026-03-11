from __future__ import annotations

from datetime import datetime, timedelta
from typing import List
import random

import numpy as np
import pandas as pd
from faker import Faker

from src.config import RAW_DIR
from src.utils import set_random_seed

fake = Faker()


INDUSTRIES: List[str] = [
    "SaaS",
    "E-commerce",
    "Marketing Agency",
    "IT Services",
    "Healthcare Services",
    "Consulting",
    "Logistics",
    "Construction",
    "Education",
    "Fintech",
]

EMAIL_DOMAINS: List[str] = [
    "gmail.com",
    "outlook.com",
    "yahoo.com",
    "companymail.com",
]

SUSPICIOUS_KEYWORDS: List[str] = [
    "global",
    "capital",
    "holdings",
    "ventures",
    "solution",
    "group",
]


def random_ip() -> str:
    return ".".join(str(random.randint(1, 255)) for _ in range(4))


def random_device_id() -> str:
    return f"dev_{fake.uuid4()[:12]}"


def clean_domain_name(company_name: str) -> str:
    cleaned = (
        company_name.lower()
        .replace(",", "")
        .replace(".", "")
        .replace("&", "and")
        .replace(" ", "")
    )
    return cleaned


def generate_business_name() -> str:
    suffixes = ["Inc", "LLC", "Corp", "Solutions", "Systems", "Labs", "Group"]
    prefix = fake.company().split(" ")[0]
    suffix = random.choice(suffixes)
    return f"{prefix} {suffix}"


def generate_domain(company_name: str, risky: bool) -> str:
    base = clean_domain_name(company_name)

    if risky and random.random() < 0.35:
        base = random.choice(SUSPICIOUS_KEYWORDS) + str(random.randint(10, 999))

    tld = random.choice([".com", ".io", ".co", ".net"])
    return f"{base}{tld}"


def generate_email(first_name: str, last_name: str, domain: str, risky: bool) -> str:
    if risky and random.random() < 0.30:
        return f"{first_name.lower()}{random.randint(10, 999)}@{random.choice(EMAIL_DOMAINS)}"
    return f"{first_name.lower()}.{last_name.lower()}@{domain.replace('www.', '')}"


def generate_incorporation_date(risky: bool, reference_time: datetime | None = None) -> datetime:
    today = reference_time or datetime.now()
    if risky:
        days_ago = random.randint(1, 120)
    else:
        days_ago = random.randint(180, 3650)
    return today - timedelta(days=days_ago)


def generate_declared_revenue(employee_count: int, risky: bool) -> float:
    if risky and random.random() < 0.4:
        return round(random.uniform(2_000_000, 20_000_000), 2)

    base = employee_count * random.uniform(20_000, 120_000)
    noise = random.uniform(0.8, 1.2)
    return round(base * noise, 2)


def generate_application_timestamp(reference_time: datetime | None = None) -> datetime:
    now = reference_time or datetime.now()
    days_back = random.randint(0, 180)
    minutes_back = random.randint(0, 1440)
    return now - timedelta(days=days_back, minutes=minutes_back)


def generate_risk_label(
    incorporation_date: datetime,
    email: str,
    declared_revenue: float,
    employee_count: int,
    reference_time: datetime | None = None,
) -> int:
    risk_score = 0

    ref_time = reference_time or datetime.now()
    business_age_days = (ref_time - incorporation_date).days
    if business_age_days < 90:
        risk_score += 2

    public_email_domains = ["gmail.com", "yahoo.com", "outlook.com"]
    if any(email.endswith(domain) for domain in public_email_domains):
        risk_score += 2

    if employee_count <= 3 and declared_revenue > 5_000_000:
        risk_score += 2

    if random.random() < 0.08:
        risk_score += 1

    return 1 if risk_score >= 3 else 0


def generate_applications(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    set_random_seed(seed)
    Faker.seed(seed)

    reference_time = datetime(2025, 1, 1, 12, 0, 0) + timedelta(seconds=seed)
    rows = []

    shared_ips = [random_ip() for _ in range(50)]
    shared_devices = [random_device_id() for _ in range(50)]

    for idx in range(1, n_rows + 1):
        risky_profile = random.random() < 0.18

        first_name = fake.first_name()
        last_name = fake.last_name()
        business_name = generate_business_name()
        domain = generate_domain(business_name, risky_profile)
        email = generate_email(first_name, last_name, domain, risky_profile)
        employee_count = random.choice([1, 2, 3, 5, 8, 15, 25, 50, 120, 250])
        incorporation_date = generate_incorporation_date(risky_profile, reference_time)
        declared_revenue = generate_declared_revenue(employee_count, risky_profile)

        if risky_profile and random.random() < 0.4:
            ip_address = random.choice(shared_ips)
            device_id = random.choice(shared_devices)
        else:
            ip_address = random_ip()
            device_id = random_device_id()

        row = {
            "application_id": f"APP_{idx:06d}",
            "business_name": business_name,
            "owner_first_name": first_name,
            "owner_last_name": last_name,
            "email": email,
            "domain": domain,
            "industry": random.choice(INDUSTRIES),
            "employee_count": employee_count,
            "declared_revenue": declared_revenue,
            "incorporation_date": incorporation_date.date().isoformat(),
            "ip_address": ip_address,
            "device_id": device_id,
            "application_timestamp": generate_application_timestamp(reference_time).isoformat(),
        }

        row["is_fraud"] = generate_risk_label(
            incorporation_date=incorporation_date,
            email=email,
            declared_revenue=declared_revenue,
            employee_count=employee_count,
            reference_time=reference_time,
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    # make fraud slightly less common and more realistic
    fraud_idx = df.index[df["is_fraud"] == 1].tolist()
    if len(fraud_idx) > int(0.12 * len(df)):
        keep_count = int(0.12 * len(df))
        flip_back = random.sample(fraud_idx, len(fraud_idx) - keep_count)
        df.loc[flip_back, "is_fraud"] = 0

    return df


def main() -> None:
    df = generate_applications(n_rows=5000, seed=42)
    output_path = RAW_DIR / "applications.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved applications data to: {output_path}")
    print(df.head())
    print("\nFraud rate:")
    print(df["is_fraud"].value_counts(normalize=True).round(4))


if __name__ == "__main__":
    main()