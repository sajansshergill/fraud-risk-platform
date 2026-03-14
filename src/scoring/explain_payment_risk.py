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


def generate_payment_risk_reasons(row: pd.Series) -> List[str]:
    reasons: List[str] = []

    amount = _safe_float(row.get("amount"))
    credit_limit = _safe_float(row.get("assigned_credit_limit"))
    amount_to_limit_ratio = _safe_float(row.get("amount_to_limit_ratio"))
    amount_zscore = _safe_float(row.get("amount_zscore_vs_account"))
    vendor_risk_score = _safe_float(row.get("vendor_risk_score"))
    xgb_score = _safe_float(row.get("xgb_score"))
    behavior_risk_score = _safe_float(row.get("behavior_risk_score"))
    payment_risk_score = _safe_float(row.get("payment_risk_score"))
    ip_risk_score = _safe_float(row.get("ip_risk_score"))
    device_risk_score = _safe_float(row.get("device_risk_score"))
    email_reputation_score = _safe_float(row.get("email_reputation_score"))

    merchant_category = str(row.get("merchant_category", "Unknown"))
    country = str(row.get("country", "Unknown"))

    if amount_to_limit_ratio >= 0.60:
        reasons.append(
            f"Transaction amount uses a large share of the credit limit ({amount_to_limit_ratio:.0%})."
        )
    elif amount_to_limit_ratio >= 0.35:
        reasons.append(
            f"Transaction amount is elevated relative to the credit limit ({amount_to_limit_ratio:.0%})."
        )

    if amount_zscore >= 3.0:
        reasons.append(
            f"Transaction amount is a strong outlier versus account history (z-score={amount_zscore:.2f})."
        )
    elif amount_zscore >= 2.0:
        reasons.append(
            f"Transaction amount is above the normal account pattern (z-score={amount_zscore:.2f})."
        )

    if _safe_int(row.get("country_risk_flag")) == 1:
        reasons.append(f"Transaction occurred in a higher-risk country ({country}).")

    if _safe_int(row.get("high_risk_merchant_flag")) == 1:
        reasons.append(f"Merchant category is higher risk ({merchant_category}).")

    if _safe_int(row.get("new_device_flag")) == 1:
        reasons.append("Transaction came from a device not seen earlier on the account.")

    if _safe_int(row.get("new_ip_flag")) == 1:
        reasons.append("Transaction came from an unfamiliar IP address for this account.")

    if ip_risk_score >= 0.70:
        reasons.append(f"IP risk score is elevated ({ip_risk_score:.2f}).")

    if device_risk_score >= 0.70:
        reasons.append(f"Device risk score is elevated ({device_risk_score:.2f}).")

    if email_reputation_score < 0.45:
        reasons.append(
            f"Associated application email reputation is weak ({email_reputation_score:.2f})."
        )

    if vendor_risk_score >= 0.65:
        reasons.append(f"Related account has a high upstream vendor risk score ({vendor_risk_score:.2f}).")

    if payment_risk_score >= 0.55:
        reasons.append(f"Rule-based payment risk score is elevated ({payment_risk_score:.2f}).")

    if behavior_risk_score >= 0.50:
        reasons.append(f"Behavioral risk score is elevated ({behavior_risk_score:.2f}).")

    if xgb_score >= 0.75:
        reasons.append(f"Internal payments model predicted high fraud probability ({xgb_score:.2f}).")

    if credit_limit > 0 and amount > 0 and amount >= 0.75 * credit_limit:
        reasons.append(
            f"Transaction amount (${amount:,.0f}) is unusually close to the assigned limit (${credit_limit:,.0f})."
        )

    if not reasons:
        reasons.append("No single dominant payment-fraud signal; score is driven by multiple moderate indicators.")

    return reasons


def generate_payment_reviewer_summary(row: pd.Series) -> str:
    final_blended_score = _safe_float(row.get("final_blended_score"))
    xgb_score = _safe_float(row.get("xgb_score"))
    vendor_risk_score = _safe_float(row.get("vendor_risk_score"))

    if final_blended_score >= 0.85:
        risk_band = "Very High Risk"
        recommendation = "Immediate investigation recommended."
    elif final_blended_score >= 0.70:
        risk_band = "High Risk"
        recommendation = "Analyst review recommended before allowing downstream action."
    elif final_blended_score >= 0.50:
        risk_band = "Moderate Risk"
        recommendation = "Additional verification may be helpful."
    else:
        risk_band = "Lower Risk"
        recommendation = "No strong payment-fraud concern from current signals."

    return (
        f"{risk_band}: blended score={final_blended_score:.2f}, "
        f"internal model score={xgb_score:.2f}, vendor score={vendor_risk_score:.2f}. "
        f"{recommendation}"
    )