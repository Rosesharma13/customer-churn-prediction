# 📉 Customer Churn Prediction

## Codec Technologies AI Internship Project

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## 📌 Objective

Predict whether a telecom customer will churn (leave the service) based on historical behavioral data. Uses classification algorithms to identify at-risk customers and key churn factors.

---

## 🛠️ Models Used

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 82.6% | 0.795 |
| Random Forest | 81.1% | 0.742 |
| XGBoost | 80.8% | 0.738 |

**Best Model: Logistic Regression (AUC: 0.795)**

---

## 📊 Key Features Used

- Customer tenure (months)
- Monthly & total charges
- Contract type (Month-to-month / One year / Two year)
- Internet service type
- Payment method
- Senior citizen status
- Tech support & online backup subscription
- Number of services

---

## 📸 Screenshots

### EDA — Churn Distribution & Patterns
<img width="1785" height="1329" alt="image" src="https://github.com/user-attachments/assets/87bd0261-7419-4282-b174-0c1a85fd0a48" />


### Model Comparison
<img width="1335" height="734" alt="image" src="https://github.com/user-attachments/assets/167fd496-a625-4f8c-8244-3330483b30af" />


### ROC Curves
<img width="1184" height="882" alt="image" src="https://github.com/user-attachments/assets/b571672f-84c3-490d-bbc1-9f8b41445383" />


### Confusion Matrix
<img width="849" height="731" alt="image" src="https://github.com/user-attachments/assets/f8ea440f-d837-449b-b478-162b77c89737" />


### Feature Importance (XGBoost)
<img width="1334" height="732" alt="image" src="https://github.com/user-attachments/assets/5c1e8237-c0fd-465e-9390-ac45dbb6802a" />

---

## 📁 Project Structure

```
customer-churn-prediction/
├── churn_prediction.py     ← Main script
├── data/
│   └── churn_data.csv      ← Generated dataset (5,000 customers)
├── screenshots/            ← All chart outputs
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
git clone https://github.com/Rosesharma13/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
python churn_prediction.py
```

---

## 📈 Key Insights

- **17.6% churn rate** in the dataset
- **Month-to-month contracts** have the highest churn risk
- **Long-tenure customers** (>24 months) are significantly less likely to churn
- **High monthly charges** correlate positively with churn
- **Tech support** subscription reduces churn probability

---

## 🔑 Key Learnings

- Handling class imbalance in churn datasets
- Feature engineering from customer behavior data
- Comparing multiple classification algorithms
- Interpreting ROC-AUC and confusion matrix for business decisions

---

## 👩‍💻 Author

**Rose Sharma** | Codec Technologies AI Internship

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rose-sharma13)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Rosesharma13)
