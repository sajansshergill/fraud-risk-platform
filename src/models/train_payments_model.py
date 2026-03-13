from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.utils import set_random_seed

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def load_training_data() -> pd.DataFrame:
    input_path: Path = PROCESSED_DIR / "payments_features.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run build_payment_features first."
        )
    return pd.read_csv(input_path)

def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    
    categorical_transformer = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    
    return ColumnTransformer(
        transformers = [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    
def main() -> None:
    set_random_seed(42)
    
    df = load_training_data()
    
    target_col = "is_fraud_transaction"
    drop_cols = ["transaction_id", "account_id", "application_id", target_col]
    
    X = df.drop(columns=drop_cols)
    y = df[target_col].astype(int)
    
    categorical_features = ["industry", "risk_tier", "merchant_category", "country"]
    numeric_features = [col for col in X.columns if col not in categorical_features]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    logistic_pipeline = Pipeline(
        steps = [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    logistic_pipeline.fit(X_train, y_train)
    
    logistic_probs = logistic_pipeline.predict_proba(X_test)[:, 1]
    logistic_preds = (logistic_probs >= 0.5).astype(int)
    
    print("\n=== Payments Logistic Regression Results ===")
    print("ROC-AUC:", round(roc_auc_score(y_test, logistic_probs), 4))
    print("PR-AUC", round(average_precision_score(y_test, logistic_probs), 4))
    print(classification_report(y_test, logistic_preds, digits=4))
    
    fit_preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_encoded = fit_preprocessor.fit_transform(X_train)
    X_test_encoded = fit_preprocessor.transform(X_test)
    
    xgb_model = XGBClassifier(
        n_estimators = 250,
        max_depth = 5,
        learning_rate = 0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=max((len(y_train) - y_train.sum()) / max(y_train.sum(), 1), 1),
    )
    xgb_model.fit(X_train_encoded, y_train)
    
    xgb_probs = xgb_model.predict_proba(X_test_encoded)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)
    
    print("\n=== Payments XGBoost Results ===")
    print("ROC-AUC:", round(roc_auc_score(y_test, xgb_probs), 4))
    print("PR-AUC :", round(average_precision_score(y_test, xgb_probs), 4))
    print(classification_report(y_test, xgb_preds, digits=4))
    
    joblib.dump(logistic_pipeline, ARTIFACT_DIR / "payments_logistic_pipeline.joblib")
    joblib.dump(fit_preprocessor, ARTIFACT_DIR / "payments_preprocessor.joblib")
    joblib.dump(xgb_model, ARTIFACT_DIR / "payments_xgb_model.joblib")
    
    scored = X_test.copy()
    scored["actual_is_fraud_transaction"] = y_test.values
    scored["logistic_score"] = logistic_probs
    scored["xgb_score"] = xgb_probs
    scored["final_blended_score"] = (
        0.35 * scored["vendor_risk_score"] +
        0.65 * scored["xgb_score"]
    ).round(4)
    
    scored = scored.sort_values("final_blended_score", ascending=False)
    output_path = PROCESSED_DIR / "payments_test_predictions.csv"
    scored.to_csv(output_path, index=False)
    
    print(f"\n Saved payments artifacts to: {ARTIFACT_DIR}")
    print(f"Saved scored transaction test set to: {output_path}")
    
if __name__ == "__main__":
    main()