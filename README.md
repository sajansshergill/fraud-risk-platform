# Ramp Fraud & Risk Intelligence Platform

A portfolio project inspired by fraud-risk workflows in B2B fintech platforms, designed to simulate how an applied science team could combine **vendor signlas, document verification, behavioral analytics, and internal ML models** to detect **underwriting fraud, payments fraud, and account takeover risks.**

--- 

## Why I Build This

I wanted to build a project that goes beyond generic fraud detection and instead focus on the kind of layered risk systems used in modern B2B spend-management and corporate card platforms.

This project simulates a real-world fraud stack where risk decisions are not made from a single model alone, but from the combination of:
- business application data
- customer-submitted documents
- third-pasrtyu vendor risk signals
- behavioral activity signals
- internal machine learning models
- explainability and ananlyst-facing review outputs

The goal is to demonstrate both:
1. **applied ML/scientific thinking**, and
2. **prodcut-oriented fraud reasoning**

---

## Problem Statement

B2B fintech platforms face fraud across multiple surfaces:

- **Underwriting Fraud**
  Fraudulent or synthetic businesses apply for spend accounts or corporate cards.
- **Payments Fraud**
  Approved accounts later exhibit suspicious or abusive transaction patterns.
- **Account Takover (ATO)**
  Legittimate accounts are compromised through suspicious login or device activity.

Fraud teams typically rely on a layered approach:
- vendor-provided risk intelligence
- internal features and heuristics
- supervised ML models
- human analyst investigation

This repository builds a simplified but realistic version of that pipeline.

---

## Project Objectives

- Simulate a B2B fraud environment using synthetic but realistic datasets
- Build risk features from applications, documents, vendor scores, and behavior logs
- Train spearate models for:
  - underwriting/application fraud
  - transaction/payment fraud
- combine vendor signals with internal model outputs into a final risk score
- generate interpretable fraud expectations using feature attribution
- provide analyst-friendly review output or high-risk cases

---

## Core Use Cases

### 1. Underwriting Fraud Detection
Given a business application, estimate whether the business is likely fraudulent using:
- business identity consistency
- document verification signals
- vendor risk scores
- IP/device reuse patterns
- domain age and email reputation

### 2. Payment Fraud Detection
Given a stream of transactions, estimate whether a transaction is suspicious using:
- amount anomalies
- merchant/category shifts
- transaction velocity
- unusual geography or device changes
- historical spend behavior

### 3. Fraud Analyst Review
Provide a ranked list of high-risk applications or transactions with explanations such as:
- domain created 4 days ago
- EIN document does not match declared business name
- multiple applications from same device
- abnormal spend spike compared to account basline

---

## End-to-End workflow

```text
Synthetic Data Generation
        ↓
Raw Data Tables
        ↓
Feature Engineering
        ↓
Vendor + Internal Risk Signals
        ↓
Model Training
        ↓
Risk Scoring Layer
        ↓
Explainability Layer
        ↓
Analyst Review Dashboard / API
```

## Dataset Design
This project uses synthetic data to simulate a B2B fintech fraus ecosystem.

### Main Tables
-> applications
Business application data for underwriting decisions.

Example fields:
- application_id
- business_name
- email
- domain
- industry
- declared_revenue
- employee_count
- incorporation_date
- ip_address
- device_id
- application_timestamp

-> documents
Submitted business verification documents.

Example fields:
- doc_id
- application_id
- document_type
- ocr_business_name
- ocr_address
- document_metadat_score
- validation_score

Examples of simulated document types:
- bank statement
- EIN letters
- certificates of incorporation

-> vendor_risk_signals
Third-partu risk vendor outputs.

Example fields:
- application_id
- domain_age_days
- email_reputation_score
- device_risk_score
- ip_risk_score
- business_registry_match_score

This simulated the "external signals" layer often used in fraud teams.

-> accounts
Approved business accounts created after underwriting.

Example fields:
- account_id
- application_id
- approval_timestamp
- assigned_credit_limit
- risk_tier

-> transactions
Card or spend transactions to business accounts.

Example fields:
- transaction_id
- account_id
- merchant_name
- merchant_category
- amount
- country
- currency
- transacion_timestamp
- device_id

-> behavior_events
Behavioral activity logs used for ATO and fraud analysis.

Example fields:
- event_id
- account_id
- event_type
- ip_address
- device_id
- geo_region
- event_timestamp

Events may include:
- login
- password reset
- MFA failure
- profile change
- payment attempt

## Feature Engineering
The project focuses on **fraud-relevant signals**, not just generic tabular modeling.

**A. Identity Consistency Features**
Used for underwriting risk.
Examples:
- email_domain_matched_business
- business_name_doc_similarity
- address_doc_similarity
- domain_age_days
- business_registry_match_score

**B. Document Verification Features**
Signals derived from comparing applications to submitted documents.

Examples:
- doc_name_mismatch_flag
- doc_address_mismatch_flag
- ocr_confidence_score
- document_metadat_anomaly_score

**C. Behavioral Features**
Used for both underwriting and ATO-like analysis

Examples:
- applications_from_same_ip_24h
- applications_from_same_device_7d
- login_velocity_score
- new_device_after_approval_flag
- geo_change_risk_score

**D. Transaction Features**
Used for payment fraud detection.

Examples:
- amount_zscore_vs_account_history
- merchant_category_shift_score
- weekend_spend_ratio
- international_transaction_flag
- transaction_velocity_1h
- merchant_entropy_30d

## Modeling Approach
This project uses a layered modeling structure instead of relying on one monolithic score.

### Model 1: Underwriting Fraud Model
Predicts probability than an application is fraudulent.

Possible algorithms:
- Logistic Regression
- Random Forest
- XGBoost

Input signals include:
- identity consistency
- document mismatch
- vendor scores
- IP/device resuse
- domain and email quality

### Model 2: Payment Fraud Model
Predicts probability that a transaction is fraudulent.

Possible algorithms:
- Logistic Regression
- Random Forest
- XGBoost

Input signals include:
- transaction anomaly scores
- merchant/ category shifts
- geography/device changes
- account spend history

## Vendor + Internal Model Layering
One of the key idea in this project is that vendor scores should not be treated as the final answer. They are useful, but the strongest system often layers internal intelligence on top.

Example combined scoring logic:
final_risk_score = (
    0.35 * vendor_risk_score +
    0.65 * internal_model_probability
)

This refelcts a more realistic fraud setup where:
- vendor tools provide broad risk coverage
- internal models adapt to platform-specific fraud patterns

## Explanability
To make model outputs actionab;e for fraud analysts, this project includes an explanability layer.

Techniques:
- SHAP values for model-level explanations
- rule-based natural language summaries

Example explanation:
- Domain is only 6 days old
- Submitted business name differs from OC-extracted document name
- Same device submitted 4 recent applications
- Vendor device risk is high

This bridges the gap between prediction and operation investigation.


## Evaluation Strategy
Since fraud is higly imbalanced, the project emphasizes metrics beyond plain accuracy.

### Metrics
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Precision@K for analyst queue usefulness

## Business-Oriented Evaluation
- false positive rate among approved accounts
- fraud capture rate in top risk decile
- investigation queue efficiency
- reduction is risk approvals under threshold policy scenarios

## Project Structure
<img width="237" height="616" alt="image" src="https://github.com/user-attachments/assets/55bc7f8e-2682-44fe-bca6-67c9e7d1abba" />

## Tech Stack
- Python
- SQL
- pandas / NumPy
- scikit-learn
- XGBoost
- SHAP
- DuckDB or Postgres
- FastAPI
- Streamlit
- Matplotlib / Plotly
- MLflow
- Docker

## Example Analyst Output
**Application Review Example**
**Application ID:** APP_10482
**Risk Score:** 0.91
**Decision Recommendation:** Manual Review

Top reasons:
- Business domain created 3 days ago
- OCR business name mismatch with application
- Device reused across 5 applications
- Vendor device risk score above threshold

**Transaction Review Example**
**Transaction ID:** TXN_88210
**Risk Score:** 0.87
**Decision Recommendation:** Flag for Investigation

Top reasons:
- Transaction amount is 9.8x normal account average
- First international spend event
- High-risk merchant category
- Login from new device within 20 minutes of transaction

## How to Run
1. Clone the repository
git clone https://github.com/yourusername/ramp-fraud-risk-intelligence.git
cd ramp-fraud-risk-intelligence

2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Generate synthetic data
python -m src.data_generation.generate_applications
python -m src.data_generation.generate_documents
python -m src.data_generation.generate_vendor_signals
python -m src.data_generation.generate_transactions
python -m src.data_generation.generate_behavior_events

5. Build features and train models
python -m src.models.train_underwriting_model
python -m src.models.train_payments_model
python -m src.models.evaluate_models

6. Launch API
uvicorn src.api.app:app --reload

7. Launch dashboard
streamlit run dashboard/streamlit_app.py

## Key Learnings This Project Demonstrates
- fraud problems are multi-layered, not single-table classification tasks
- third-party risk signals become stronger when combined with platform-specific features
- interpretable fraud systems are more useful than black-box risk scores
- good applied science work connects modeling to operational decision-making
