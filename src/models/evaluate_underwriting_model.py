from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT


ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def precision_at_k(df: pd.DataFrame, score_col: str, target_col: str, k: int) -> float:
    top_k = df.sort_values(score_col, ascending=False).head(k)
    if len(top_k) == 0:
        return 0.0
    return float(top_k[target_col].mean())


def fraud_capture_at_k(df: pd.DataFrame, score_col: str, target_col: str, k: int) -> float:
    total_frauds = max(int(df[target_col].sum()), 1)
    top_k = df.sort_values(score_col, ascending=False).head(k)
    captured = int(top_k[target_col].sum())
    return captured / total_frauds


def main() -> None:
    predictions_path: Path = ARTIFACT_DIR / "underwriting_test_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"{predictions_path} not found. Run train_underwriting_model first."
        )

    df = pd.read_csv(predictions_path)

    target_col = "actual_is_fraud"
    score_cols = ["logistic_score", "xgb_score", "final_blended_score"]

    print("\n=== Underwriting Model Queue Quality ===")
    for score_col in score_cols:
        print(f"\n--- {score_col} ---")
        for k in [25, 50, 100, 200]:
            p_at_k = precision_at_k(df, score_col, target_col, k)
            c_at_k = fraud_capture_at_k(df, score_col, target_col, k)
            print(
                f"K={k:<3} | Precision@K={p_at_k:.4f} | Fraud Capture@K={c_at_k:.4f}"
            )

    print("\nTop 15 highest-risk applications by blended score:")
    display_cols = [
        "industry",
        "employee_count",
        "declared_revenue",
        "business_age_days",
        "vendor_risk_score",
        "doc_risk_score",
        "identity_risk_score",
        "behavior_risk_score",
        "xgb_score",
        "final_blended_score",
        "actual_is_fraud",
    ]
    print(df[display_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()