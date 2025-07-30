# Task 3: Model Explainability with SHAP
# This script generates SHAP explanations for the best model (Random Forest) on both datasets.

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Helper to load and preprocess data (same as in Task 2)
def load_creditcard_data():
    df = pd.read_csv('creditcard.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

def load_fraud_data():
    df = pd.read_csv('Fraud_Data.csv')
    df.drop_duplicates(inplace=True)
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['user_transaction_count'] = df.groupby('user_id')['purchase_time'].transform('count')
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    X = df.drop(['class', 'signup_time', 'purchase_time', 'ip_address', 'user_id', 'device_id'], axis=1)
    y = df['class']
    X = pd.get_dummies(X, columns=['browser', 'source', 'sex'], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train_res, y_test, X.columns.tolist()

# Helper to train and return the best model (Random Forest)
def train_best_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# SHAP explainability for a dataset
def shap_explain_dataset(dataset_name, load_data_func):
    print(f'\n===== SHAP Explainability for {dataset_name} =====')
    X_train, X_test, y_train, y_test, feature_names = load_data_func()
    model = train_best_rf(X_train, y_train)
    # Use a sample for SHAP to save time/memory
    X_explain = X_test[:2000] if X_test.shape[0] > 2000 else X_test
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)
    # Summary plot (global feature importance)
    plt.figure()
    shap.summary_plot(shap_values[1], X_explain, feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot - {dataset_name}')
    plt.savefig(f'shap_summary_{dataset_name}.png', bbox_inches='tight', dpi=300)
    plt.close()
    # Force plot (local explanation for a single instance)
    force_plot_path = f'shap_force_{dataset_name}.png'
    shap.initjs()
    # Use the first fraud case if possible, else first instance
    idx = np.where(y_test.values[:len(X_explain)] == 1)[0]
    i = idx[0] if len(idx) > 0 else 0
    force_plot = shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][i],
        X_explain[i],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f'SHAP Force Plot - {dataset_name} (Instance {i})')
    plt.savefig(force_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'  Saved summary plot: shap_summary_{dataset_name}.png')
    print(f'  Saved force plot:   shap_force_{dataset_name}.png')
    # Return top features for report
    shap_abs = np.abs(shap_values[1]).mean(axis=0)
    top_idx = np.argsort(shap_abs)[::-1][:10]
    top_features = [(feature_names[j], shap_abs[j]) for j in top_idx]
    return top_features

def main():
    print('TASK 3: Model Explainability with SHAP')
    print('='*60)
    # Credit Card Dataset
    cc_top_features = shap_explain_dataset('creditcard', load_creditcard_data)
    # Fraud Data Dataset
    fd_top_features = shap_explain_dataset('fraud_data', load_fraud_data)
    # Write markdown report
    with open('task3_shap_report.md', 'w') as f:
        f.write('# Task 3: Model Explainability with SHAP\n')
        f.write('\n## Credit Card Dataset\n')
        f.write('**Top 10 SHAP Features (Global Importance):**\n')
        for name, val in cc_top_features:
            f.write(f'- {name}: {val:.4f}\n')
        f.write('\n![](shap_summary_creditcard.png)\n')
        f.write('\n![](shap_force_creditcard.png)\n')
        f.write('\nInterpretation: The summary plot shows which features most influence the model globally. The force plot explains a single prediction (likely a fraud case).\n')
        f.write('\n## Fraud Data Dataset\n')
        f.write('**Top 10 SHAP Features (Global Importance):**\n')
        for name, val in fd_top_features:
            f.write(f'- {name}: {val:.4f}\n')
        f.write('\n![](shap_summary_fraud_data.png)\n')
        f.write('\n![](shap_force_fraud_data.png)\n')
        f.write('\nInterpretation: The summary plot shows the most important drivers of fraud in this dataset. The force plot explains a single prediction.\n')
        f.write('\n## General Interpretation\n')
        f.write('SHAP summary plots reveal the most influential features for fraud detection. Features at the top have the greatest impact on the model’s output. Force plots show how individual feature values push a prediction toward fraud or not fraud.\n')
    print('SHAP report and plots saved. See task3_shap_report.md for interpretation.')

if __name__ == '__main__':
    main()