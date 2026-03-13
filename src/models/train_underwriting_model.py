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


MODEL_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data() -> pd.DataFrame:
    input_path: Path = PROCESSED_DIR / "underwriting_features.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run build_underwriting_features first."
        )
    return pd.read_csv(input_path)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def train_logistic_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    pipeline = Pipeline(
        steps=[
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

    pipeline.fit(X_train, y_train)
    return pipeline


def train_xgboost_model(
    X_train_encoded,
    y_train: pd.Series,
) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=max((len(y_train) - y_train.sum()) / max(y_train.sum(), 1), 1),
    )
    model.fit(X_train_encoded, y_train)
    return model


def main() -> None:
    set_random_seed(42)

    df = load_training_data()

    target_col = "is_fraud"
    id_col = "application_id"
    app_ids = df[id_col].astype(str)
    drop_cols = [id_col, target_col]

    X = df.drop(columns=drop_cols)
    y = df[target_col].astype(int)

    categorical_features = ["industry"]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    test_app_ids = app_ids.loc[X_test.index]

    logistic_pipeline = train_logistic_pipeline(
        X_train=X_train,
        y_train=y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    logistic_probs = logistic_pipeline.predict_proba(X_test)[:, 1]
    logistic_preds = (logistic_probs >= 0.5).astype(int)

    print("\n=== Logistic Regression Results ===")
    print("ROC-AUC:", round(roc_auc_score(y_test, logistic_probs), 4))
    print("PR-AUC :", round(average_precision_score(y_test, logistic_probs), 4))
    print(classification_report(y_test, logistic_preds, digits=4))

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    xgb_model = train_xgboost_model(X_train_encoded=X_train_encoded, y_train=y_train)
    xgb_probs = xgb_model.predict_proba(X_test_encoded)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)

    print("\n=== XGBoost Results ===")
    print("ROC-AUC:", round(roc_auc_score(y_test, xgb_probs), 4))
    print("PR-AUC :", round(average_precision_score(y_test, xgb_probs), 4))
    print(classification_report(y_test, xgb_preds, digits=4))

    joblib.dump(logistic_pipeline, MODEL_DIR / "underwriting_logistic_pipeline.joblib")
    joblib.dump(preprocessor, MODEL_DIR / "underwriting_preprocessor.joblib")
    joblib.dump(xgb_model, MODEL_DIR / "underwriting_xgb_model.joblib")

    test_predictions = X_test.copy()
    test_predictions[id_col] = test_app_ids.values
    test_predictions["actual_is_fraud"] = y_test.values
    test_predictions["logistic_score"] = logistic_probs
    test_predictions["xgb_score"] = xgb_probs
    test_predictions["final_blended_score"] = (
        0.40 * test_predictions["vendor_risk_score"]
        + 0.60 * test_predictions["xgb_score"]
    ).round(4)

    test_predictions = test_predictions.sort_values(
        by="final_blended_score", ascending=False
    )
    predictions_path = MODEL_DIR / "underwriting_test_predictions.csv"
    test_predictions.to_csv(predictions_path, index=False)

    print(f"\nSaved models to: {MODEL_DIR}")
    print(f"Saved scored test set to: {predictions_path}")


if __name__ == "__main__":
    main()
