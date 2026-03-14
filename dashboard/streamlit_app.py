from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.scoring.explain_payment_risk import (
    generate_payment_reasons := None,  # placeholder to keep linters calm if needed
)
from src.scoring.explain_payment_risk import (
    generate_payment_reviewer_summary,
    generate_payment_risk_reasons,
)
from src.scoring.explain_risk import generate_reviewer_summary, generate_risk_reasons


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

UNDERWRITING_PATH = ARTIFACT_DIR / "underwriting_test_predictions.csv"
PAYMENTS_PATH = ARTIFACT_DIR / "payments_test_predictions.csv"


st.set_page_config(
    page_title="Ramp Fraud Analyst Dashboard",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def load_underwriting_predictions() -> pd.DataFrame:
    if not UNDERWRITING_PATH.exists():
        raise FileNotFoundError(
            f"{UNDERWRITING_PATH} not found. Run the underwriting pipeline first."
        )

    df = pd.read_csv(UNDERWRITING_PATH)

    df["risk_band"] = pd.cut(
        df["final_blended_score"],
        bins=[-0.01, 0.50, 0.70, 0.85, 1.00],
        labels=["Lower", "Moderate", "High", "Very High"],
    )

    df["review_recommendation"] = df["final_blended_score"].apply(
        lambda x: (
            "Strong Manual Review"
            if x >= 0.85
            else "Manual Review"
            if x >= 0.70
            else "Additional Verification"
            if x >= 0.50
            else "Auto-Approve Candidate"
        )
    )

    return df.sort_values("final_blended_score", ascending=False).reset_index(drop=True)


@st.cache_data
def load_payments_predictions() -> pd.DataFrame:
    if not PAYMENTS_PATH.exists():
        raise FileNotFoundError(
            f"{PAYMENTS_PATH} not found. Run the payments pipeline first."
        )

    df = pd.read_csv(PAYMENTS_PATH)

    df["risk_band"] = pd.cut(
        df["final_blended_score"],
        bins=[-0.01, 0.50, 0.70, 0.85, 1.00],
        labels=["Lower", "Moderate", "High", "Very High"],
    )

    df["review_recommendation"] = df["final_blended_score"].apply(
        lambda x: (
            "Immediate Investigation"
            if x >= 0.85
            else "Fraud Analyst Review"
            if x >= 0.70
            else "Additional Verification"
            if x >= 0.50
            else "Likely Normal"
        )
    )

    return df.sort_values("final_blended_score", ascending=False).reset_index(drop=True)


def format_currency(value: float) -> str:
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "$0"


def render_underwriting_header(df: pd.DataFrame) -> None:
    total_cases = len(df)
    actual_fraud_rate = float(df["actual_is_fraud"].mean()) if total_cases else 0.0
    avg_score = float(df["final_blended_score"].mean()) if total_cases else 0.0
    high_risk_cases = int((df["final_blended_score"] >= 0.85).sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Scored Applications", f"{total_cases:,}")
    col2.metric("Actual Fraud Rate", f"{actual_fraud_rate:.1%}")
    col3.metric("Avg Blended Risk", f"{avg_score:.2f}")
    col4.metric("Very High Risk Queue", f"{high_risk_cases:,}")


def render_payments_header(df: pd.DataFrame) -> None:
    total_cases = len(df)
    actual_fraud_rate = (
        float(df["actual_is_fraud_transaction"].mean()) if total_cases else 0.0
    )
    avg_score = float(df["final_blended_score"].mean()) if total_cases else 0.0
    high_risk_cases = int((df["final_blended_score"] >= 0.85).sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Scored Transactions", f"{total_cases:,}")
    col2.metric("Actual Fraud Rate", f"{actual_fraud_rate:.1%}")
    col3.metric("Avg Blended Risk", f"{avg_score:.2f}")
    col4.metric("Very High Risk Queue", f"{high_risk_cases:,}")


def render_underwriting_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Underwriting Filters")

    min_score = st.sidebar.slider(
        "Minimum underwriting risk score",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        key="uw_min_score",
    )

    risk_bands = st.sidebar.multiselect(
        "Underwriting risk bands",
        options=["Lower", "Moderate", "High", "Very High"],
        default=["Moderate", "High", "Very High"],
        key="uw_risk_bands",
    )

    industries = sorted(df["industry"].dropna().astype(str).unique().tolist())
    selected_industries = st.sidebar.multiselect(
        "Underwriting industries",
        options=industries,
        default=industries,
        key="uw_industries",
    )

    review_options = sorted(
        df["review_recommendation"].dropna().astype(str).unique().tolist()
    )
    selected_review_options = st.sidebar.multiselect(
        "Underwriting recommendation",
        options=review_options,
        default=review_options,
        key="uw_review_options",
    )

    return df[
        (df["final_blended_score"] >= min_score)
        & (df["risk_band"].astype(str).isin(risk_bands))
        & (df["industry"].astype(str).isin(selected_industries))
        & (df["review_recommendation"].astype(str).isin(selected_review_options))
    ].copy()


def render_payments_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Payments Filters")

    min_score = st.sidebar.slider(
        "Minimum payments risk score",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        key="p_min_score",
    )

    risk_bands = st.sidebar.multiselect(
        "Payments risk bands",
        options=["Lower", "Moderate", "High", "Very High"],
        default=["Moderate", "High", "Very High"],
        key="p_risk_bands",
    )

    merchant_categories = sorted(
        df["merchant_category"].dropna().astype(str).unique().tolist()
    )
    selected_categories = st.sidebar.multiselect(
        "Merchant categories",
        options=merchant_categories,
        default=merchant_categories,
        key="p_categories",
    )

    countries = sorted(df["country"].dropna().astype(str).unique().tolist())
    selected_countries = st.sidebar.multiselect(
        "Countries",
        options=countries,
        default=countries,
        key="p_countries",
    )

    review_options = sorted(
        df["review_recommendation"].dropna().astype(str).unique().tolist()
    )
    selected_review_options = st.sidebar.multiselect(
        "Payments recommendation",
        options=review_options,
        default=review_options,
        key="p_review_options",
    )

    return df[
        (df["final_blended_score"] >= min_score)
        & (df["risk_band"].astype(str).isin(risk_bands))
        & (df["merchant_category"].astype(str).isin(selected_categories))
        & (df["country"].astype(str).isin(selected_countries))
        & (df["review_recommendation"].astype(str).isin(selected_review_options))
    ].copy()


def render_underwriting_queue_table(df: pd.DataFrame) -> None:
    st.subheader("Underwriting Review Queue")

    table_df = df[
        [
            "application_id",
            "industry",
            "employee_count",
            "declared_revenue",
            "business_age_days",
            "vendor_risk_score",
            "xgb_score",
            "final_blended_score",
            "risk_band",
            "review_recommendation",
            "actual_is_fraud",
        ]
    ].copy()

    table_df["declared_revenue"] = table_df["declared_revenue"].apply(format_currency)
    table_df["vendor_risk_score"] = table_df["vendor_risk_score"].round(3)
    table_df["xgb_score"] = table_df["xgb_score"].round(3)
    table_df["final_blended_score"] = table_df["final_blended_score"].round(3)

    st.dataframe(table_df, use_container_width=True, hide_index=True)


def render_payments_queue_table(df: pd.DataFrame) -> None:
    st.subheader("Payments Fraud Queue")

    table_df = df[
        [
            "transaction_id",
            "account_id",
            "merchant_category",
            "country",
            "amount",
            "assigned_credit_limit",
            "vendor_risk_score",
            "xgb_score",
            "final_blended_score",
            "risk_band",
            "review_recommendation",
            "actual_is_fraud_transaction",
        ]
    ].copy()

    table_df["amount"] = table_df["amount"].apply(format_currency)
    table_df["assigned_credit_limit"] = table_df["assigned_credit_limit"].apply(
        format_currency
    )
    table_df["vendor_risk_score"] = table_df["vendor_risk_score"].round(3)
    table_df["xgb_score"] = table_df["xgb_score"].round(3)
    table_df["final_blended_score"] = table_df["final_blended_score"].round(3)

    st.dataframe(table_df, use_container_width=True, hide_index=True)


def render_underwriting_case_details(df: pd.DataFrame) -> None:
    st.subheader("Underwriting Investigation")

    application_ids = df["application_id"].tolist()
    if not application_ids:
        st.warning("No underwriting applications match the selected filters.")
        return

    selected_app_id = st.selectbox(
        "Select an application",
        options=application_ids,
        index=0,
        key="uw_select",
    )

    row = df.loc[df["application_id"] == selected_app_id].iloc[0]

    summary = generate_reviewer_summary(row)
    reasons = generate_risk_reasons(row)

    left, right = st.columns([1.0, 1.3])

    with left:
        st.markdown("### Case Snapshot")
        st.write(f"**Application ID:** {row['application_id']}")
        st.write(f"**Industry:** {row['industry']}")
        st.write(f"**Employee Count:** {int(row['employee_count'])}")
        st.write(f"**Declared Revenue:** {format_currency(row['declared_revenue'])}")
        st.write(f"**Business Age:** {int(row['business_age_days'])} days")
        st.write(f"**Risk Band:** {row['risk_band']}")
        st.write(f"**Recommendation:** {row['review_recommendation']}")
        st.write(f"**Actual Label:** {int(row['actual_is_fraud'])}")

        st.markdown("### Scores")
        st.write(f"**Vendor Risk Score:** {float(row['vendor_risk_score']):.3f}")
        st.write(f"**Internal Model Score:** {float(row['xgb_score']):.3f}")
        st.write(f"**Blended Final Score:** {float(row['final_blended_score']):.3f}")

    with right:
        st.markdown("### Reviewer Summary")
        st.info(summary)

        st.markdown("### Key Risk Drivers")
        for reason in reasons:
            st.write(f"- {reason}")

        st.markdown("### Raw Signal View")
        raw_signals = {
            "domain_age_days": int(row.get("domain_age_days", 0)),
            "email_reputation_score": round(
                float(row.get("email_reputation_score", 0.0)), 3
            ),
            "ip_risk_score": round(float(row.get("ip_risk_score", 0.0)), 3),
            "device_risk_score": round(float(row.get("device_risk_score", 0.0)), 3),
            "business_registry_match_score": round(
                float(row.get("business_registry_match_score", 0.0)), 3
            ),
            "doc_name_mismatch_flag": int(row.get("doc_name_mismatch_flag", 0)),
            "doc_address_mismatch_flag": int(row.get("doc_address_mismatch_flag", 0)),
            "applications_per_ip": int(row.get("applications_per_ip", 0)),
            "applications_per_device": int(row.get("applications_per_device", 0)),
            "doc_risk_score": round(float(row.get("doc_risk_score", 0.0)), 3),
            "identity_risk_score": round(
                float(row.get("identity_risk_score", 0.0)), 3
            ),
            "behavior_risk_score": round(
                float(row.get("behavior_risk_score", 0.0)), 3
            ),
        }
        st.json(raw_signals)


def render_payments_case_details(df: pd.DataFrame) -> None:
    st.subheader("Payments Investigation")

    transaction_ids = df["transaction_id"].tolist()
    if not transaction_ids:
        st.warning("No payment transactions match the selected filters.")
        return

    selected_txn_id = st.selectbox(
        "Select a transaction",
        options=transaction_ids,
        index=0,
        key="p_select",
    )

    row = df.loc[df["transaction_id"] == selected_txn_id].iloc[0]

    summary = generate_payment_reviewer_summary(row)
    reasons = generate_payment_risk_reasons(row)

    left, right = st.columns([1.0, 1.3])

    with left:
        st.markdown("### Transaction Snapshot")
        st.write(f"**Transaction ID:** {row['transaction_id']}")
        st.write(f"**Account ID:** {row['account_id']}")
        st.write(f"**Merchant Category:** {row['merchant_category']}")
        st.write(f"**Country:** {row['country']}")
        st.write(f"**Amount:** {format_currency(row['amount'])}")
        st.write(
            f"**Assigned Credit Limit:** {format_currency(row['assigned_credit_limit'])}"
        )
        st.write(f"**Risk Band:** {row['risk_band']}")
        st.write(f"**Recommendation:** {row['review_recommendation']}")
        st.write(f"**Actual Label:** {int(row['actual_is_fraud_transaction'])}")

        st.markdown("### Scores")
        st.write(f"**Vendor Risk Score:** {float(row['vendor_risk_score']):.3f}")
        st.write(f"**Internal Model Score:** {float(row['xgb_score']):.3f}")
        st.write(f"**Blended Final Score:** {float(row['final_blended_score']):.3f}")

    with right:
        st.markdown("### Reviewer Summary")
        st.info(summary)

        st.markdown("### Key Risk Drivers")
        for reason in reasons:
            st.write(f"- {reason}")

        st.markdown("### Raw Signal View")
        raw_signals = {
            "amount_to_limit_ratio": round(
                float(row.get("amount_to_limit_ratio", 0.0)), 3
            ),
            "amount_zscore_vs_account": round(
                float(row.get("amount_zscore_vs_account", 0.0)), 3
            ),
            "country_risk_flag": int(row.get("country_risk_flag", 0)),
            "high_risk_merchant_flag": int(row.get("high_risk_merchant_flag", 0)),
            "new_device_flag": int(row.get("new_device_flag", 0)),
            "new_ip_flag": int(row.get("new_ip_flag", 0)),
            "vendor_risk_score": round(float(row.get("vendor_risk_score", 0.0)), 3),
            "ip_risk_score": round(float(row.get("ip_risk_score", 0.0)), 3),
            "device_risk_score": round(float(row.get("device_risk_score", 0.0)), 3),
            "behavior_risk_score": round(
                float(row.get("behavior_risk_score", 0.0)), 3
            ),
            "payment_risk_score": round(float(row.get("payment_risk_score", 0.0)), 3),
        }
        st.json(raw_signals)


def render_underwriting_patterns(df: pd.DataFrame) -> None:
    st.subheader("Underwriting Queue Patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Industry Breakdown")
        industry_counts = (
            df.groupby("industry", dropna=False)["application_id"]
            .count()
            .sort_values(ascending=False)
            .reset_index(name="applications")
        )
        st.dataframe(industry_counts, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Recommendation Breakdown")
        recommendation_counts = (
            df.groupby("review_recommendation", dropna=False)["application_id"]
            .count()
            .sort_values(ascending=False)
            .reset_index(name="applications")
        )
        st.dataframe(recommendation_counts, use_container_width=True, hide_index=True)


def render_payments_patterns(df: pd.DataFrame) -> None:
    st.subheader("Payments Queue Patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Merchant Category Breakdown")
        merchant_counts = (
            df.groupby("merchant_category", dropna=False)["transaction_id"]
            .count()
            .sort_values(ascending=False)
            .reset_index(name="transactions")
        )
        st.dataframe(merchant_counts, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Recommendation Breakdown")
        recommendation_counts = (
            df.groupby("review_recommendation", dropna=False)["transaction_id"]
            .count()
            .sort_values(ascending=False)
            .reset_index(name="transactions")
        )
        st.dataframe(recommendation_counts, use_container_width=True, hide_index=True)


def render_underwriting_tab() -> None:
    df = load_underwriting_predictions()
    render_underwriting_header(df)
    st.markdown("---")
    filtered_df = render_underwriting_filters(df)
    render_underwriting_queue_table(filtered_df)
    st.markdown("---")
    render_underwriting_case_details(filtered_df)
    st.markdown("---")
    render_underwriting_patterns(filtered_df)


def render_payments_tab() -> None:
    df = load_payments_predictions()
    render_payments_header(df)
    st.markdown("---")
    filtered_df = render_payments_filters(df)
    render_payments_queue_table(filtered_df)
    st.markdown("---")
    render_payments_case_details(filtered_df)
    st.markdown("---")
    render_payments_patterns(filtered_df)


def main() -> None:
    st.title("Ramp Fraud Analyst Dashboard")
    st.caption(
        "Prototype risk platform for underwriting and payments fraud using vendor signals, document checks, behavioral features, and internal ML models."
    )

    underwriting_tab, payments_tab = st.tabs(
        ["Underwriting Risk Queue", "Payments Fraud Queue"]
    )

    with underwriting_tab:
        render_underwriting_tab()

    with payments_tab:
        render_payments_tab()


if __name__ == "__main__":
    main()