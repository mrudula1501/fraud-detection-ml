# 💳 Fraud Detection ML System

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5-FF6600?style=flat)](https://xgboost.readthedocs.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.40-6366F1?style=flat)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-10B981?style=flat)](LICENSE)

Machine learning fraud detection system achieving **99.92% ROC-AUC** on 6.3M+ financial transactions. Combines XGBoost, LightGBM, and Logistic Regression in a stacking ensemble with SMOTE for class imbalance.

---

## 🎯 Problem Statement

Financial fraud costs institutions over **$32 billion annually**. Traditional rule-based systems fail because they:

- Generate 90%+ false positives, alienating legitimate customers
- Miss novel fraud patterns (low recall on unseen schemes)
- Can't scale to millions of transactions per day

This system uses an ML ensemble to achieve near-perfect AUC while remaining interpretable through SHAP explanations.

---

## 📊 Dataset

| Metric | Value |
|--------|-------|
| Total Transactions | 6,362,620 |
| Fraudulent | 8,213 (0.129%) |
| Legitimate | 6,354,407 (99.871%) |
| Features | 11 |
| Time Period | 30 days simulated |

> **Source**: [Kaggle — Online Payments Fraud Detection](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION                               │
│  Real-time transaction stream (Kafka) OR Batch CSV              │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                            │
│  • Transaction velocity (amount / time since last)              │
│  • Balance change ratios (new/old)                              │
│  • Hour-of-day / day-of-week cyclical encoding                  │
│  • Merchant category risk scores                                │
│  • Customer behavioral patterns (rolling averages)              │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL ENSEMBLE                                │
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │   XGBoost   │   │  LightGBM   │   │  Logistic   │           │
│  │  (Primary)  │   │ (Secondary) │   │  Regression │           │
│  │             │   │             │   │ (Baseline)  │           │
│  │ • SMOTE     │   │ • SMOTE     │   │ • SMOTE     │           │
│  │ • scale_pos │   │ • class_wt  │   │ • class_wt  │           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│         └─────────────────┼─────────────────┘                   │
│                           ▼                                     │
│                ┌─────────────────────┐                         │
│                │  STACKING           │                         │
│                │  Meta-learner:      │                         │
│                │  XGBoost            │                         │
│                └─────────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Fraud Score (0–1)  →  Alert (if >0.7)  →  Case Management     │
│  Threshold: 0.7 (optimized for F-beta=2)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Model Performance

### Final Results (Test Set)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | **0.9992** | Near-perfect discrimination |
| Precision | 0.96 | 96% of flagged cases are real fraud |
| Recall | 0.94 | Catches 94% of all fraud |
| F1-Score | 0.95 | Balanced precision-recall |
| **F2-Score** | **0.945** | Optimized for recall (fraud focus) |
| Average Precision | 0.98 | Excellent ranking quality |

### Confusion Matrix

```
              Predicted
           Fraud    Legit
Actual
Fraud       1,542      98    ← 94% recall
Legit          64  1,270,296 ← 99.995% specificity
```

### Baseline Comparison

| Model | ROC-AUC | Recall | Precision |
|-------|---------|--------|-----------|
| Random Guess | 0.50 | 0.50 | 0.001 |
| Logistic Regression | 0.89 | 0.72 | 0.85 |
| Random Forest | 0.97 | 0.85 | 0.91 |
| XGBoost (tuned) | 0.9992 | 0.94 | 0.96 |
| **Ensemble (XGB+LGBM+LR)** | **0.9992** | **0.94** | **0.96** |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/mrudula1501/fraud-detection-ml.git
cd fraud-detection-ml

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Download dataset
kaggle datasets download -d rupakroy/online-payments-fraud-detection-dataset
unzip online-payments-fraud-detection-dataset.zip -d data/
```

### Usage

```python
from src.fraud_detector import FraudDetector

detector = FraudDetector(model_path='models/xgboost_fraud_v1.pkl')

transaction = {
    'step': 100,
    'type': 'TRANSFER',
    'amount': 181.00,
    'oldbalanceOrg': 181.00,
    'newbalanceOrig': 0.00,
    'oldbalanceDest': 0.00,
    'newbalanceDest': 0.00
}

result = detector.predict(transaction)
print(result)
# {
#   'is_fraud': True,
#   'fraud_probability': 0.987,
#   'confidence': 'HIGH',
#   'explanation': {
#     'top_features': ['amount', 'balance_change_ratio', 'type_TRANSFER'],
#     'shap_values': [...]
#   }
# }
```

### Batch Processing

```python
import pandas as pd

df = pd.read_csv('data/transactions.csv')
predictions = detector.predict_batch(df, batch_size=10000)
predictions.to_csv('fraud_predictions.csv', index=False)
print(f"Flagged {predictions['is_fraud'].sum()} potential fraud cases")
```

### Model Training

```bash
python train.py \
  --data data/PS_20174392719_1491204439457_log.csv \
  --model xgboost \
  --tune \
  --output models/

python evaluate.py --model models/xgboost_fraud_v1.pkl --test data/test.csv
```

---

## 🔍 Feature Engineering

| Feature | Formula | Importance |
|---------|---------|------------|
| balance_change_ratio | (new − old) / old | 0.23 |
| amount_velocity | amount / hours_since_last_txn | 0.19 |
| type_TRANSFER | One-hot encoding | 0.15 |
| hour_of_day | sin/cos encoding | 0.12 |
| merchant_risk_score | Historical fraud rate | 0.11 |
| customer_txn_count_24h | Rolling window count | 0.08 |

---

## ⚖️ Handling Class Imbalance

With only 0.129% fraud, a naive model predicting "all legitimate" gets 99.87% accuracy but catches **0% of fraud**. We solve this with:

| Technique | Tool | Effect |
|-----------|------|--------|
| SMOTE | imbalanced-learn | Synthetic minority oversampling |
| scale_pos_weight | XGBoost param | Cost-sensitive learning |
| Threshold tuning | F-beta optimization | Recall-focused decisions |
| Stacking ensemble | Meta-learner | Reduced variance |

> We optimize for **F2-Score** (weights recall 2x over precision) because missing fraud costs far more than a false alarm.

---

## 📁 Project Structure

```
fraud-detection-ml/
├── data/
│   ├── raw/                   # Original Kaggle dataset
│   └── processed/             # Feature-engineered data
├── src/
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_xgboost.py
│   │   ├── train_lightgbm.py
│   │   └── ensemble.py
│   ├── explainability/
│   │   └── shap_explainer.py
│   └── fraud_detector.py      # Main API class
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_explainability.ipynb
├── models/
│   └── xgboost_fraud_v1.pkl
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔮 Future Enhancements

- [ ] Graph Neural Networks for transaction network modeling
- [ ] Neo4j for real-time relationship analysis
- [ ] AutoML with Featuretools for automated feature engineering
- [ ] Federated learning across institutions without data sharing
- [ ] Isolation Forest for unsupervised anomaly detection

---

## 📚 References

- Kaggle Dataset: https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset
- Chen & Guestrin, 2016. *XGBoost: A scalable tree boosting system.*
- Chawla et al., 2002. *SMOTE: Synthetic minority over-sampling technique.*
- Lundberg & Lee, 2017. *A unified approach to interpreting model predictions.*

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

> ⚠️ **Disclaimer**: Research project. Not for production financial use without proper compliance review.

---

## 📬 Contact

**Mrudula Deshmukh**

[![GitHub](https://img.shields.io/badge/GitHub-mrudula1501-181717?style=flat&logo=github)](https://github.com/mrudula1501)
[![Portfolio](https://img.shields.io/badge/Portfolio-mrudula1501.github.io-10B981?style=flat&logo=githubpages&logoColor=white)](https://mrudula1501.github.io/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-dmrudula-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/dmrudula/)
[![Email](https://img.shields.io/badge/Email-mrudulad25@gmail.com-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:mrudulad25@gmail.com)
