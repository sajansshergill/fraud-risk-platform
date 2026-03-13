from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.explain_risk import generate_reviewer_summary, generate_risk_reasons
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
PREDICTIONS_PATH = ARTIFACT_DIR / "underwriting_test_predictions.csv"


st.set_page_config(
    page_title="Ramp Fraud Analyst Dashboard",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{PREDICTIONS_PATH} not found. Run train_underwriting_model first."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    if "application_id" not in df.columns:
        df.insert(
            0,
            "application_id",
            [f"APP_{i+1:06d}" for i in range(len(df))],
        )

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


def format_currency(value: float) -> str:
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "$0"


def render_header(df: pd.DataFrame) -> None:
    st.title("Ramp Fraud Analyst Investigation Dashboard")
    st.caption(
        "Underwriting fraud prototype combining vendor signals, document checks, identity features, and internal model scores."
    )

    total_cases = len(df)
    actual_fraud_rate = float(df["actual_is_fraud"].mean()) if total_cases else 0.0
    avg_score = float(df["final_blended_score"].mean()) if total_cases else 0.0
    very_high_risk_cases = int((df["final_blended_score"] >= 0.85).sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Scored Applications", f"{total_cases:,}")
    col2.metric("Actual Fraud Rate", f"{actual_fraud_rate:.1%}")
    col3.metric("Avg Blended Risk", f"{avg_score:.2f}")
    col4.metric("Very High Risk Queue", f"{very_high_risk_cases:,}")


def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    min_score = st.sidebar.slider(
        "Minimum blended risk score",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
    )

    risk_bands = st.sidebar.multiselect(
        "Risk bands",
        options=["Lower", "Moderate", "High", "Very High"],
        default=["Moderate", "High", "Very High"],
    )

    industries = sorted(df["industry"].dropna().astype(str).unique().tolist())
    selected_industries = st.sidebar.multiselect(
        "Industries",
        options=industries,
        default=industries,
    )

    review_options = sorted(df["review_recommendation"].dropna().astype(str).unique().tolist())
    selected_review_options = st.sidebar.multiselect(
        "Review recommendation",
        options=review_options,
        default=review_options,
    )

    filtered = df[
        (df["final_blended_score"] >= min_score)
        & (df["risk_band"].astype(str).isin(risk_bands))
        & (df["industry"].astype(str).isin(selected_industries))
        & (df["review_recommendation"].astype(str).isin(selected_review_options))
    ].copy()

    return filtered


def render_queue_table(df: pd.DataFrame) -> None:
    st.subheader("Risk Review Queue")

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


def render_case_details(df: pd.DataFrame) -> None:
    st.subheader("Application Investigation")

    application_ids = df["application_id"].tolist()
    if not application_ids:
        st.warning("No applications match the selected filters.")
        return

    selected_app_id = st.selectbox(
        "Select an application to investigate",
        options=application_ids,
        index=0,
    )

    row = df.loc[df["application_id"] == selected_app_id].iloc[0]

    summary = generate_reviewer_summary(row)
    reasons = generate_risk_reasons(row)

    left, right = st.columns([1.1, 1.3])

    with left:
        st.markdown("### Case Snapshot")
        st.write(f"**Application ID:** {row['application_id']}")
        st.write(f"**Industry:** {row['industry']}")
        st.write(f"**Employee Count:** {int(row['employee_count'])}")
        st.write(f"**Declared Revenue:** {format_currency(row['declared_revenue'])}")
        st.write(f"**Business Age:** {int(row['business_age_days'])} days")
        st.write(f"**Risk Band:** {row['risk_band']}")
        st.write(f"**Recommendation:** {row['review_recommendation']}")
        st.write(f"**Actual Label (synthetic):** {int(row['actual_is_fraud'])}")

        st.markdown("### Model Scores")
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
            "email_reputation_score": round(float(row.get("email_reputation_score", 0.0)), 3),
            "ip_risk_score": round(float(row.get("ip_risk_score", 0.0)), 3),
            "device_risk_score": round(float(row.get("device_risk_score", 0.0)), 3),
            "business_registry_match_score": round(float(row.get("business_registry_match_score", 0.0)), 3),
            "doc_name_mismatch_flag": int(row.get("doc_name_mismatch_flag", 0)),
            "doc_address_mismatch_flag": int(row.get("doc_address_mismatch_flag", 0)),
            "applications_per_ip": int(row.get("applications_per_ip", 0)),
            "applications_per_device": int(row.get("applications_per_device", 0)),
            "doc_risk_score": round(float(row.get("doc_risk_score", 0.0)), 3),
            "identity_risk_score": round(float(row.get("identity_risk_score", 0.0)), 3),
            "behavior_risk_score": round(float(row.get("behavior_risk_score", 0.0)), 3),
        }
        st.json(raw_signals)


def render_top_patterns(df: pd.DataFrame) -> None:
    st.subheader("Queue Patterns")

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


def main() -> None:
    df = load_predictions()
    render_header(df)

    filtered_df = render_filters(df)

    st.markdown("---")
    render_queue_table(filtered_df)

    st.markdown("---")
    render_case_details(filtered_df)

    st.markdown("---")
    render_top_patterns(filtered_df)


if __name__ == "__main__":
    main()