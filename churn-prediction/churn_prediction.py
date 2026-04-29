"""
Customer Churn Prediction
Codec Technologies AI Internship Project
Author: Rose Sharma

Predicts whether a telecom customer will churn based on historical data.
Models: Logistic Regression, Random Forest, XGBoost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from xgboost import XGBClassifier
import warnings
import os
warnings.filterwarnings('ignore')

np.random.seed(42)
os.makedirs('screenshots', exist_ok=True)
os.makedirs('data', exist_ok=True)

PALETTE = ["#1a3c6e", "#e74c3c", "#27ae60", "#f39c12"]

# ─── Generate realistic telecom churn dataset ────────────────────
def generate_dataset(n=5000):
    tenure        = np.random.randint(1, 72, n)
    monthly_charge= np.round(np.random.uniform(20, 120, n), 2)
    total_charges = np.round(tenure * monthly_charge + np.random.normal(0, 50, n), 2)
    total_charges = np.clip(total_charges, 0, None)

    contract    = np.random.choice(['Month-to-month','One year','Two year'], n, p=[0.55, 0.25, 0.20])
    internet    = np.random.choice(['DSL','Fiber optic','No'], n, p=[0.35, 0.45, 0.20])
    payment     = np.random.choice(['Electronic check','Mailed check','Bank transfer','Credit card'], n)
    senior      = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner     = np.random.choice([0, 1], n, p=[0.52, 0.48])
    dependents  = np.random.choice([0, 1], n, p=[0.70, 0.30])
    tech_support= np.random.choice([0, 1], n, p=[0.50, 0.50])
    online_backup=np.random.choice([0, 1], n, p=[0.50, 0.50])
    num_services= np.random.randint(1, 7, n)

    # Churn probability based on realistic factors
    churn_prob = (
        0.35 * (contract == 'Month-to-month').astype(float) +
        0.15 * (internet == 'Fiber optic').astype(float) +
        0.10 * senior +
        0.08 * (payment == 'Electronic check').astype(float) -
        0.15 * (tenure > 24).astype(float) -
        0.10 * tech_support -
        0.05 * online_backup +
        0.02 * (monthly_charge > 80).astype(float) +
        np.random.normal(0, 0.08, n)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = (np.random.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        'tenure': tenure,
        'MonthlyCharges': monthly_charge,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet,
        'PaymentMethod': payment,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'TechSupport': tech_support,
        'OnlineBackup': online_backup,
        'NumServices': num_services,
        'Churn': churn
    })
    return df


# ─── Preprocessing ───────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    le = LabelEncoder()
    for col in ['Contract', 'InternetService', 'PaymentMethod']:
        df[col] = le.fit_transform(df[col])
    return df


# ─── Plotting ────────────────────────────────────────────────────
def plot_eda(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Customer Churn — Exploratory Data Analysis', fontsize=14, fontweight='bold', y=1.01)

    # Churn distribution
    churn_counts = df['Churn'].value_counts()
    axes[0,0].pie(churn_counts, labels=['Retained','Churned'], autopct='%1.1f%%',
                  colors=[PALETTE[0], PALETTE[1]], startangle=90, textprops={'fontsize':11})
    axes[0,0].set_title('Churn Distribution', fontweight='bold')

    # Tenure by churn
    df[df['Churn']==0]['tenure'].hist(ax=axes[0,1], bins=20, alpha=0.7, color=PALETTE[0], label='Retained')
    df[df['Churn']==1]['tenure'].hist(ax=axes[0,1], bins=20, alpha=0.7, color=PALETTE[1], label='Churned')
    axes[0,1].set_title('Tenure Distribution by Churn', fontweight='bold')
    axes[0,1].set_xlabel('Tenure (months)')
    axes[0,1].legend()

    # Monthly charges by churn
    df.boxplot(column='MonthlyCharges', by='Churn', ax=axes[1,0],
               boxprops=dict(color=PALETTE[0]), medianprops=dict(color=PALETTE[1]))
    axes[1,0].set_title('Monthly Charges by Churn', fontweight='bold')
    axes[1,0].set_xlabel('Churn (0=No, 1=Yes)')
    plt.sca(axes[1,0])
    plt.title('Monthly Charges by Churn')

    # Contract type vs churn
    contract_churn = df.groupby('Contract')['Churn'].mean()
    axes[1,1].bar(range(len(contract_churn)), contract_churn.values,
                  color=[PALETTE[0], PALETTE[2], PALETTE[3]])
    axes[1,1].set_xticks(range(len(contract_churn)))
    axes[1,1].set_xticklabels(['Month-to-month','One year','Two year'], rotation=10)
    axes[1,1].set_title('Churn Rate by Contract Type', fontweight='bold')
    axes[1,1].set_ylabel('Churn Rate')

    plt.tight_layout()
    plt.savefig('screenshots/eda.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ EDA chart saved")


def plot_model_comparison(results):
    names = list(results.keys())
    accs  = [results[n]['accuracy'] for n in names]
    aucs  = [results[n]['roc_auc'] for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - 0.2, accs, 0.35, label='Accuracy', color=PALETTE[0])
    bars2 = ax.bar(x + 0.2, aucs, 0.35, label='ROC-AUC', color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0.5, 1.0)
    ax.set_title('Model Comparison — Accuracy & ROC-AUC', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.spines[['top','right']].set_visible(False)
    for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{bar.get_height():.3f}', ha='center', fontsize=9)
    for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{bar.get_height():.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('screenshots/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Model comparison chart saved")


def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Retained','Churned'], yticklabels=['Retained','Churned'])
    ax.set_title(f'Confusion Matrix — {model_name}', fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('screenshots/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrix saved")


def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(idx)), importance[idx], color=PALETTE[0])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_title('Feature Importance — XGBoost', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig('screenshots/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Feature importance saved")


def plot_roc_curve(models_roc):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [PALETTE[0], PALETTE[2], PALETTE[1]]
    for i, (name, (fpr, tpr, auc)) in enumerate(models_roc.items()):
        ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig('screenshots/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ ROC curves saved")


# ─── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("CUSTOMER CHURN PREDICTION — Codec Technologies")
    print("=" * 55)

    # Generate and save data
    print("\n📊 Generating dataset...")
    df = generate_dataset(5000)
    df.to_csv('data/churn_data.csv', index=False)
    print(f"✅ Dataset: {len(df):,} customers | Churn rate: {df['Churn'].mean():.1%}")

    # EDA
    plot_eda(df)

    # Preprocess
    df_processed = preprocess(df)
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train models
    print("\n🤖 Training models...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost':             XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)
    }

    results   = {}
    models_roc= {}

    for name, model in models.items():
        X_tr = X_train_sc if name == 'Logistic Regression' else X_train
        X_te = X_test_sc  if name == 'Logistic Regression' else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:,1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        results[name]    = {'accuracy': acc, 'roc_auc': auc}
        models_roc[name] = (fpr, tpr, auc)
        print(f"  {name:<25} Acc: {acc:.3f}  AUC: {auc:.3f}")

    # Best model
    best_name  = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = models[best_name]
    best_X_te  = X_test_sc if best_name == 'Logistic Regression' else X_test
    y_pred_best= best_model.predict(best_X_te)

    print(f"\n🏆 Best Model: {best_name}")
    print(f"\n{classification_report(y_test, y_pred_best, target_names=['Retained','Churned'])}")

    # Plots
    print("\n📊 Generating visualizations...")
    plot_model_comparison(results)
    plot_confusion_matrix(confusion_matrix(y_test, y_pred_best), best_name)
    plot_roc_curve(models_roc)

    xgb = models['XGBoost']
    plot_feature_importance(xgb, list(X.columns))

    print("\n" + "=" * 55)
    print(f"✅ COMPLETE — Best: {best_name} | AUC: {results[best_name]['roc_auc']:.3f}")
    print("=" * 55)
