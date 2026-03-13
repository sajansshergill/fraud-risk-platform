from __future__ import annotations

from typing import List

import pandas as pd


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def generate_risk_reasons(row: pd.Series) -> List[str]:
    reasons: List[str] = []

    business_age_days = _safe_int(row.get("business_age_days"))
    domain_age_days = _safe_int(row.get("domain_age_days"))
    vendor_risk_score = _safe_float(row.get("vendor_risk_score"))
    xgb_score = _safe_float(row.get("xgb_score"))
    doc_risk_score = _safe_float(row.get("doc_risk_score"))
    identity_risk_score = _safe_float(row.get("identity_risk_score"))
    behavior_risk_score = _safe_float(row.get("behavior_risk_score"))
    email_reputation_score = _safe_float(row.get("email_reputation_score"))
    ip_risk_score = _safe_float(row.get("ip_risk_score"))
    device_risk_score = _safe_float(row.get("device_risk_score"))
    registry_match_score = _safe_float(row.get("business_registry_match_score"))
    applications_per_ip = _safe_int(row.get("applications_per_ip"), 1)
    applications_per_device = _safe_int(row.get("applications_per_device"), 1)
    employee_count = _safe_int(row.get("employee_count"))
    declared_revenue = _safe_float(row.get("declared_revenue"))

    if domain_age_days > 0 and domain_age_days < 30:
        reasons.append(f"Business domain is very new ({domain_age_days} days old).")
    elif domain_age_days > 0 and domain_age_days < 90:
        reasons.append(f"Business domain is relatively new ({domain_age_days} days old).")

    if business_age_days > 0 and business_age_days < 90:
        reasons.append(f"Business entity appears newly incorporated ({business_age_days} days old).")

    if _safe_int(row.get("doc_name_mismatch_flag")) == 1:
        reasons.append("Submitted document business name does not fully match application data.")

    if _safe_int(row.get("doc_address_mismatch_flag")) == 1:
        reasons.append("Submitted document address shows mismatch with expected business information.")

    if email_reputation_score < 0.45:
        reasons.append(f"Email reputation score is low ({email_reputation_score:.2f}).")

    if _safe_int(row.get("is_public_email_domain")) == 1:
        reasons.append("Applicant used a public email domain instead of a company domain.")

    if _safe_int(row.get("domain_email_match_flag")) == 0:
        reasons.append("Applicant email domain does not match the declared company domain.")

    if ip_risk_score > 0.70:
        reasons.append(f"IP risk score is elevated ({ip_risk_score:.2f}).")

    if device_risk_score > 0.70:
        reasons.append(f"Device risk score is elevated ({device_risk_score:.2f}).")

    if registry_match_score < 0.60:
        reasons.append(
            f"Business registry match score is low ({registry_match_score:.2f}), indicating identity inconsistency."
        )

    if applications_per_ip >= 3:
        reasons.append(f"IP address has been used across {applications_per_ip} applications.")

    if applications_per_device >= 3:
        reasons.append(f"Device has been used across {applications_per_device} applications.")

    if employee_count <= 3 and declared_revenue > 5_000_000:
        reasons.append(
            f"Declared revenue (${declared_revenue:,.0f}) is unusually high for a company with {employee_count} employees."
        )

    if doc_risk_score >= 0.45:
        reasons.append(f"Document risk score is elevated ({doc_risk_score:.2f}).")

    if identity_risk_score >= 0.45:
        reasons.append(f"Identity risk score is elevated ({identity_risk_score:.2f}).")

    if behavior_risk_score >= 0.45:
        reasons.append(f"Behavior risk score is elevated ({behavior_risk_score:.2f}).")

    if vendor_risk_score >= 0.65:
        reasons.append(f"Third-party vendor risk score is high ({vendor_risk_score:.2f}).")

    if xgb_score >= 0.70:
        reasons.append(f"Internal model predicted high underwriting fraud probability ({xgb_score:.2f}).")

    if not reasons:
        reasons.append("No major single risk driver identified; score is based on a combination of moderate signals.")

    return reasons


def generate_reviewer_summary(row: pd.Series) -> str:
    final_blended_score = _safe_float(row.get("final_blended_score"))
    xgb_score = _safe_float(row.get("xgb_score"))
    vendor_risk_score = _safe_float(row.get("vendor_risk_score"))

    if final_blended_score >= 0.85:
        risk_band = "Very High Risk"
        recommendation = "Strong manual review recommended before approval."
    elif final_blended_score >= 0.70:
        risk_band = "High Risk"
        recommendation = "Manual review recommended."
    elif final_blended_score >= 0.50:
        risk_band = "Moderate Risk"
        recommendation = "Additional verification may be useful."
    else:
        risk_band = "Lower Risk"
        recommendation = "No severe concerns from current underwriting signals."

    summary = (
        f"{risk_band}: blended score={final_blended_score:.2f}, "
        f"internal model score={xgb_score:.2f}, vendor score={vendor_risk_score:.2f}. "
        f"{recommendation}"
    )
    return summary