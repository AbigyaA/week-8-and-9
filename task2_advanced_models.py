# Task 2 - Advanced Model Building and Training
# Enhanced implementation with XGBoost, LightGBM, and hyperparameter tuning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score, f1_score,
    roc_curve, auc, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

# Set random seed for reproducibility
np.random.seed(42)

class AdvancedFraudDetectionModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        self.best_params = {}
        
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
        
        # Data preprocessing
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
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for each model"""
        print(f"\n{'='*50}")
        print(f"HYPERPARAMETER TUNING FOR {self.dataset_name.upper()}")
        print(f"{'='*50}")
        
        # 1. Logistic Regression tuning
        print("\n1. Tuning Logistic Regression...")
        lr_param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        lr_grid = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            lr_param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        lr_grid.fit(X_train, y_train)
        self.best_params['logistic_regression'] = lr_grid.best_params_
        print(f"Best LR params: {lr_grid.best_params_}")
        
        # 2. Random Forest tuning
        print("\n2. Tuning Random Forest...")
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            rf_param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        self.best_params['random_forest'] = rf_grid.best_params_
        print(f"Best RF params: {rf_grid.best_params_}")
        
        # 3. Gradient Boosting tuning
        print("\n3. Tuning Gradient Boosting...")
        gb_param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            gb_param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        gb_grid.fit(X_train, y_train)
        self.best_params['gradient_boosting'] = gb_grid.best_params_
        print(f"Best GB params: {gb_grid.best_params_}")
        
        # 4. XGBoost tuning (if available)
        if XGBOOST_AVAILABLE:
            print("\n4. Tuning XGBoost...")
            xgb_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            xgb_grid = GridSearchCV(
                xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                xgb_param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            xgb_grid.fit(X_train, y_train)
            self.best_params['xgboost'] = xgb_grid.best_params_
            print(f"Best XGB params: {xgb_grid.best_params_}")
        
        # 5. LightGBM tuning (if available)
        if LIGHTGBM_AVAILABLE:
            print("\n5. Tuning LightGBM...")
            lgb_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            lgb_grid = GridSearchCV(
                lgb.LGBMClassifier(random_state=42, verbose=-1),
                lgb_param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            lgb_grid.fit(X_train, y_train)
            self.best_params['lightgbm'] = lgb_grid.best_params_
            print(f"Best LGB params: {lgb_grid.best_params_}")
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train all models with optimized parameters"""
        print(f"\n{'='*50}")
        print(f"TRAINING OPTIMIZED MODELS FOR {self.dataset_name.upper()}")
        print(f"{'='*50}")
        
        # 1. Logistic Regression
        print("\n1. Training optimized Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            **self.best_params['logistic_regression']
        )
        lr_model.fit(X_train, y_train)
        self.models['logistic_regression'] = lr_model
        
        # 2. Random Forest
        print("2. Training optimized Random Forest...")
        rf_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **self.best_params['random_forest']
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # 3. Gradient Boosting
        print("3. Training optimized Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            random_state=42,
            **self.best_params['gradient_boosting']
        )
        gb_model.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb_model
        
        # 4. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            print("4. Training optimized XGBoost...")
            xgb_model = xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                **self.best_params['xgboost']
            )
            xgb_model.fit(X_train, y_train)
            self.models['xgboost'] = xgb_model
        
        # 5. LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            print("5. Training optimized LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                random_state=42,
                verbose=-1,
                **self.best_params['lightgbm']
            )
            lgb_model.fit(X_train, y_train)
            self.models['lightgbm'] = lgb_model
        
        print("All optimized models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models using comprehensive metrics"""
        print(f"\n{'='*50}")
        print(f"COMPREHENSIVE MODEL EVALUATION FOR {self.dataset_name.upper()}")
        print(f"{'='*50}")
        
        for model_name, model in self.models.items():
            print(f"\n{'-'*40}")
            print(f"EVALUATING {model_name.upper()}")
            print(f"{'-'*40}")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate comprehensive metrics
            results = {
                'accuracy': (y_pred == y_test).mean(),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba),
                'auc_pr': average_precision_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            self.results[model_name] = results
            
            # Print detailed results
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"F1-Score: {results['f1_score']:.4f}")
            print(f"AUC-ROC: {results['auc_roc']:.4f}")
            print(f"AUC-PR: {results['auc_pr']:.4f}")
            print(f"\nConfusion Matrix:")
            print(results['confusion_matrix'])
            print(f"\nClassification Report:")
            print(results['classification_report'])
    
    def plot_comprehensive_results(self, y_test):
        """Create comprehensive visualization of all results"""
        print(f"\n{'='*50}")
        print(f"CREATING COMPREHENSIVE VISUALIZATIONS FOR {self.dataset_name.upper()}")
        print(f"{'='*50}")
        
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Advanced Model Performance Comparison - {self.dataset_name.upper()}', fontsize=16)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # 1. ROC Curves
        ax1 = axes[0]
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
        ax2 = axes[1]
        for model_name, results in self.results.items():
            precision, recall, _ = precision_recall_curve(y_test, results['y_pred_proba'])
            ax2.plot(recall, precision, label=f'{model_name} (AUC-PR = {results["auc_pr"]:.3f})')
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Metrics Comparison Bar Chart
        ax3 = axes[2]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(metrics))
        width = 0.8 / len(self.models)
        
        for i, (model_name, results) in enumerate(self.results.items()):
            values = [results[metric] for metric in metrics]
            ax3.bar(x + i * width, values, width, label=model_name, alpha=0.8)
        
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Score')
        ax3.set_title('Metrics Comparison')
        ax3.set_xticks(x + width * (len(self.models) - 1) / 2)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4-6. Confusion Matrices
        for i, (model_name, results) in enumerate(self.results.items()):
            if i + 3 < len(axes):
                ax = axes[i + 3]
                cm = results['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'{model_name.title()} Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(len(self.models) + 3, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.dataset_name}_advanced_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def comprehensive_model_comparison(self):
        """Comprehensive model comparison and analysis"""
        print(f"\n{'='*50}")
        print(f"COMPREHENSIVE MODEL COMPARISON FOR {self.dataset_name.upper()}")
        print(f"{'='*50}")
        
        # Create comprehensive comparison dataframe
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'AUC-ROC': results['auc_roc'],
                'AUC-PR': results['auc_pr']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nComprehensive Model Performance Comparison:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Determine best model based on different criteria
        best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
        best_auc_pr = comparison_df.loc[comparison_df['AUC-PR'].idxmax(), 'Model']
        best_auc_roc = comparison_df.loc[comparison_df['AUC-ROC'].idxmax(), 'Model']
        
        print(f"\nBest Models by Different Criteria:")
        print(f"  - Best F1-Score: {best_f1}")
        print(f"  - Best AUC-PR: {best_auc_pr}")
        print(f"  - Best AUC-ROC: {best_auc_roc}")
        
        # Detailed analysis
        print(f"\nDetailed Model Analysis:")
        for model_name, results in self.results.items():
            print(f"\n{model_name.upper()}:")
            if model_name == 'logistic_regression':
                print(f"  - Strengths: Highly interpretable, fast training, good baseline")
                print(f"  - Weaknesses: Limited to linear relationships")
            elif model_name == 'random_forest':
                print(f"  - Strengths: Handles non-linear relationships, feature importance")
                print(f"  - Weaknesses: Can be prone to overfitting")
            elif model_name == 'gradient_boosting':
                print(f"  - Strengths: Often achieves high performance, handles outliers well")
                print(f"  - Weaknesses: Sensitive to hyperparameters, slower training")
            elif model_name == 'xgboost':
                print(f"  - Strengths: Excellent performance, handles missing values")
                print(f"  - Weaknesses: Requires careful tuning, can be complex")
            elif model_name == 'lightgbm':
                print(f"  - Strengths: Fast training, memory efficient, good performance")
                print(f"  - Weaknesses: May be less robust to outliers")
            
            print(f"  - F1-Score: {results['f1_score']:.4f}")
            print(f"  - AUC-PR: {results['auc_pr']:.4f}")
            print(f"  - AUC-ROC: {results['auc_roc']:.4f}")
        
        # Recommendation
        print(f"\nRECOMMENDATION:")
        print(f"For {self.dataset_name.upper()} dataset:")
        print(f"  - Best overall model: {best_f1} (based on F1-Score)")
        print(f"  - Best for precision-focused tasks: {best_auc_pr}")
        print(f"  - Best for balanced performance: {best_auc_roc}")
        
        return best_f1

def main():
    """Main function to run the advanced analysis"""
    print("TASK 2: ADVANCED MODEL BUILDING AND TRAINING")
    print("="*70)
    
    # Analyze both datasets
    datasets = ['creditcard', 'fraud_data']
    best_models = {}
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"ADVANCED ANALYSIS OF {dataset.upper()} DATASET")
        print(f"{'='*70}")
        
        # Initialize advanced model class
        fraud_model = AdvancedFraudDetectionModel(dataset)
        
        # Load and prepare data
        X_train, X_test, y_train, y_test = fraud_model.load_and_prepare_data()
        
        # Perform hyperparameter tuning
        fraud_model.hyperparameter_tuning(X_train, y_train)
        
        # Train optimized models
        fraud_model.train_models(X_train, X_test, y_train, y_test)
        
        # Evaluate models
        fraud_model.evaluate_models(X_test, y_test)
        
        # Plot comprehensive results
        fraud_model.plot_comprehensive_results(y_test)
        
        # Comprehensive model comparison
        best_model = fraud_model.comprehensive_model_comparison()
        best_models[dataset] = best_model
        
        print(f"\n{'='*70}")
        print(f"FINAL SUMMARY FOR {dataset.upper()}")
        print(f"{'='*70}")
        print(f"Best Model: {best_model}")
        print(f"Key Metrics for Best Model:")
        best_results = fraud_model.results[best_model]
        print(f"  - F1-Score: {best_results['f1_score']:.4f}")
        print(f"  - AUC-PR: {best_results['auc_pr']:.4f}")
        print(f"  - AUC-ROC: {best_results['auc_roc']:.4f}")
        print(f"  - Precision: {best_results['precision']:.4f}")
        print(f"  - Recall: {best_results['recall']:.4f}")
    
    # Final comparison across datasets
    print(f"\n{'='*70}")
    print("FINAL COMPARISON ACROSS DATASETS")
    print(f"{'='*70}")
    for dataset, best_model in best_models.items():
        print(f"{dataset.upper()}: Best Model = {best_model}")
    
    print(f"\n{'='*70}")
    print("ADVANCED TASK 2 COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main() 