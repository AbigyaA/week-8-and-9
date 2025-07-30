# Task 2 - Model Building and Training
# Comprehensive implementation for both Credit Card and Fraud Data datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score, f1_score,
    roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class FraudDetectionModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data based on dataset type"""
        if self.dataset_name == 'creditcard':
            return self._prepare_creditcard_data()
        elif self.dataset_name == 'fraud_data':
            return self._prepare_fraud_data()
        else:
            raise ValueError("Dataset must be 'creditcard' or 'fraud_data'")
    
    def _prepare_creditcard_data(self):
        """Prepare Credit Card dataset"""
        print(f"\n{'='*50}")
        print(f"PREPARING {self.dataset_name.upper()} DATASET")
        print(f"{'='*50}")
        
        # Load data
        df = pd.read_csv('creditcard.csv')
        print(f"Original dataset shape: {df.shape}")
        
        # Check for missing values
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Check class distribution
        print(f"Class distribution:\n{y.value_counts(normalize=True)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _prepare_fraud_data(self):
        """Prepare Fraud Data dataset"""
        print(f"\n{'='*50}")
        print(f"PREPARING {self.dataset_name.upper()} DATASET")
        print(f"{'='*50}")
        
        # Load data
        df = pd.read_csv('Fraud_Data.csv')
        print(f"Original dataset shape: {df.shape}")
        
        # Check for missing values
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Data preprocessing (similar to task1.py)
        df.drop_duplicates(inplace=True)
        
        # Convert datetime fields
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        # Feature engineering
        df['user_transaction_count'] = df.groupby('user_id')['purchase_time'].transform('count')
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        
        # Prepare features and target
        X = df.drop(['class', 'signup_time', 'purchase_time', 'ip_address', 'user_id', 'device_id'], axis=1)
        y = df['class']
        
        # Encode categorical features
        X = pd.get_dummies(X, columns=['browser', 'source', 'sex'], drop_first=True)
        
        # Check class distribution
        print(f"Class distribution:\n{y.value_counts(normalize=True)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_resampled)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set shape (after SMOTE): {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train_resampled, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train Logistic Regression and Random Forest models"""
        print(f"\n{'='*50}")
        print(f"TRAINING MODELS FOR {self.dataset_name.upper()}")
        print(f"{'='*50}")
        
        # 1. Logistic Regression
        print("\n1. Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        self.models['logistic_regression'] = lr_model
        
        # 2. Random Forest
        print("2. Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        print("Model training completed!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models using multiple metrics"""
        print(f"\n{'='*50}")
        print(f"MODEL EVALUATION FOR {self.dataset_name.upper()}")
        print(f"{'='*50}")
        
        for model_name, model in self.models.items():
            print(f"\n{'-'*30}")
            print(f"EVALUATING {model_name.upper()}")
            print(f"{'-'*30}")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            results = {
                'accuracy': (y_pred == y_test).mean(),
                'f1_score': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba),
                'auc_pr': average_precision_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            self.results[model_name] = results
            
            # Print results
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"F1-Score: {results['f1_score']:.4f}")
            print(f"AUC-ROC: {results['auc_roc']:.4f}")
            print(f"AUC-PR: {results['auc_pr']:.4f}")
            print(f"\nConfusion Matrix:")
            print(results['confusion_matrix'])
            print(f"\nClassification Report:")
            print(results['classification_report'])
    
    def plot_results(self, y_test):
        """Create comprehensive visualization of results"""
        print(f"\n{'='*50}")
        print(f"CREATING VISUALIZATIONS FOR {self.dataset_name.upper()}")
        print(f"{'='*50}")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Performance Comparison - {self.dataset_name.upper()}', fontsize=16)
        
        # 1. ROC Curves
        ax1 = axes[0, 0]
        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
            ax1.plot(fpr, tpr, label=f'{model_name} (AUC = {results["auc_roc"]:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Precision-Recall Curves
        ax2 = axes[0, 1]
        for model_name, results in self.results.items():
            precision, recall, _ = precision_recall_curve(y_test, results['y_pred_proba'])
            ax2.plot(recall, precision, label=f'{model_name} (AUC-PR = {results["auc_pr"]:.3f})')
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Confusion Matrices
        for i, (model_name, results) in enumerate(self.results.items()):
            ax = axes[1, i]
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model_name.title()} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{self.dataset_name}_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self):
        """Compare models and identify the best one"""
        print(f"\n{'='*50}")
        print(f"MODEL COMPARISON FOR {self.dataset_name.upper()}")
        print(f"{'='*50}")
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'F1-Score': results['f1_score'],
                'AUC-ROC': results['auc_roc'],
                'AUC-PR': results['auc_pr']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Determine best model based on F1-Score and AUC-PR (important for imbalanced data)
        best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
        print(f"\nBest Model (based on F1-Score): {best_model}")
        
        # Additional analysis
        print(f"\nDetailed Analysis:")
        for model_name, results in self.results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  - Strengths: Good for interpretability" if model_name == 'logistic_regression' else "  - Strengths: Handles non-linear relationships well")
            print(f"  - F1-Score: {results['f1_score']:.4f}")
            print(f"  - AUC-PR: {results['auc_pr']:.4f}")
        
        return best_model

def main():
    """Main function to run the complete analysis"""
    print("TASK 2: MODEL BUILDING AND TRAINING")
    print("="*60)
    
    # Analyze both datasets
    datasets = ['creditcard', 'fraud_data']
    best_models = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"ANALYZING {dataset.upper()} DATASET")
        print(f"{'='*60}")
        
        # Initialize model class
        fraud_model = FraudDetectionModel(dataset)
        
        # Load and prepare data
        X_train, X_test, y_train, y_test = fraud_model.load_and_prepare_data()
        
        # Train models
        fraud_model.train_models(X_train, X_test, y_train, y_test)
        
        # Evaluate models
        fraud_model.evaluate_models(X_test, y_test)
        
        # Plot results
        fraud_model.plot_results(y_test)
        
        # Compare models and get best one
        best_model = fraud_model.compare_models()
        best_models[dataset] = best_model
        
        print(f"\n{'='*60}")
        print(f"SUMMARY FOR {dataset.upper()}")
        print(f"{'='*60}")
        print(f"Best Model: {best_model}")
        print(f"Key Metrics for Best Model:")
        best_results = fraud_model.results[best_model]
        print(f"  - F1-Score: {best_results['f1_score']:.4f}")
        print(f"  - AUC-PR: {best_results['auc_pr']:.4f}")
        print(f"  - AUC-ROC: {best_results['auc_roc']:.4f}")
    
    # Final comparison across datasets
    print(f"\n{'='*60}")
    print("FINAL COMPARISON ACROSS DATASETS")
    print(f"{'='*60}")
    for dataset, best_model in best_models.items():
        print(f"{dataset.upper()}: Best Model = {best_model}")
    
    print(f"\n{'='*60}")
    print("TASK 2 COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 