# Task 2: Model Building and Training

## Overview
This implementation provides comprehensive model building and training for fraud detection using both the Credit Card and Fraud Data datasets. The solution includes both basic and advanced implementations with multiple machine learning models and evaluation metrics.

## Files Structure

```
week-8-and-9/
├── task2_model_building.py          # Basic implementation
├── task2_advanced_models.py         # Advanced implementation with hyperparameter tuning
├── run_task2.py                     # Execution script
├── requirements.txt                 # Dependencies
├── README_Task2.md                 # This file
├── creditcard.csv                  # Credit card fraud dataset
├── Fraud_Data.csv                  # Fraud data dataset
└── IpAddress_to_Country.csv        # IP address mapping
```

## Implementation Details

### 1. Data Preparation

#### Credit Card Dataset (`creditcard.csv`)
- **Target Variable**: `Class` (0 = Normal, 1 = Fraud)
- **Features**: 30 anonymized features (V1-V28, Amount, Time)
- **Preprocessing**: 
  - Standard scaling of features
  - Train-test split (80-20) with stratification
  - No resampling needed (handles imbalance through metrics)

#### Fraud Data Dataset (`Fraud_Data.csv`)
- **Target Variable**: `class` (0 = Not Fraud, 1 = Fraud)
- **Features**: User demographics, transaction details, device info
- **Preprocessing**:
  - Feature engineering (time-based features, user transaction count)
  - Categorical encoding (browser, source, sex)
  - SMOTE resampling for class imbalance
  - Standard scaling

### 2. Model Selection

#### Basic Implementation (`task2_model_building.py`)
1. **Logistic Regression**: Simple, interpretable baseline model
2. **Random Forest**: Ensemble model for non-linear relationships

#### Advanced Implementation (`task2_advanced_models.py`)
1. **Logistic Regression**: With hyperparameter tuning
2. **Random Forest**: With hyperparameter tuning
3. **Gradient Boosting**: Additional ensemble method
4. **XGBoost**: High-performance gradient boosting (if available)
5. **LightGBM**: Fast gradient boosting (if available)

### 3. Model Training and Evaluation

#### Training Process
- **Train-Test Split**: 80-20 split with stratification
- **Cross-Validation**: 5-fold CV for hyperparameter tuning
- **Class Imbalance Handling**: SMOTE for Fraud Data dataset

#### Evaluation Metrics
For imbalanced data, we use appropriate metrics:

1. **F1-Score**: Harmonic mean of precision and recall
2. **AUC-PR (Average Precision)**: Area under Precision-Recall curve
3. **AUC-ROC**: Area under ROC curve
4. **Precision**: True positives / (True positives + False positives)
5. **Recall**: True positives / (True positives + False negatives)
6. **Confusion Matrix**: Detailed breakdown of predictions

### 4. Hyperparameter Tuning

#### Logistic Regression
- `C`: Regularization strength [0.001, 0.01, 0.1, 1, 10, 100]
- `penalty`: Regularization type ['l1', 'l2']
- `solver`: Optimization algorithm ['liblinear', 'saga']

#### Random Forest
- `n_estimators`: Number of trees [50, 100, 200]
- `max_depth`: Maximum tree depth [5, 10, 15, None]
- `min_samples_split`: Minimum samples to split [2, 5, 10]
- `min_samples_leaf`: Minimum samples per leaf [1, 2, 4]

#### Gradient Boosting
- `n_estimators`: Number of boosting stages [50, 100, 200]
- `learning_rate`: Learning rate [0.01, 0.1, 0.2]
- `max_depth`: Maximum tree depth [3, 5, 7]
- `subsample`: Fraction of samples for fitting [0.8, 0.9, 1.0]

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Running the Implementation

#### Basic Implementation
```bash
python run_task2.py basic
# or
python task2_model_building.py
```

#### Advanced Implementation
```bash
python run_task2.py advanced
# or
python task2_advanced_models.py
```

### Direct Execution
```bash
# Basic implementation
python task2_model_building.py

# Advanced implementation
python task2_advanced_models.py
```

## Output

### Console Output
- Dataset information and preprocessing steps
- Model training progress
- Detailed evaluation metrics for each model
- Model comparison and recommendations

### Visualizations
- ROC curves for all models
- Precision-Recall curves
- Confusion matrices
- Metrics comparison bar charts
- Saved as PNG files: `{dataset}_model_comparison.png`

### Key Results
- Best model identification based on F1-Score
- Comprehensive performance metrics
- Model strengths and weaknesses analysis
- Recommendations for each dataset

## Model Justification

### Why These Models?

1. **Logistic Regression**
   - **Strengths**: Highly interpretable, fast training, good baseline
   - **Use Case**: Baseline model for comparison, when interpretability is crucial

2. **Random Forest**
   - **Strengths**: Handles non-linear relationships, feature importance, robust
   - **Use Case**: When you need feature importance and good performance

3. **Gradient Boosting**
   - **Strengths**: Often achieves highest performance, handles outliers well
   - **Use Case**: When maximum performance is required

4. **XGBoost**
   - **Strengths**: Excellent performance, handles missing values, regularization
   - **Use Case**: Production systems requiring high accuracy

5. **LightGBM**
   - **Strengths**: Fast training, memory efficient, good performance
   - **Use Case**: Large datasets or when speed is important

### Best Model Selection Criteria

For fraud detection with imbalanced data, we prioritize:

1. **F1-Score**: Primary metric (harmonic mean of precision and recall)
2. **AUC-PR**: Important for imbalanced datasets
3. **AUC-ROC**: Overall model performance
4. **Precision**: Minimizing false positives (important for fraud detection)

## Expected Results

### Credit Card Dataset
- **Class Distribution**: Highly imbalanced (~0.17% fraud)
- **Expected Best Model**: Usually XGBoost or Random Forest
- **Expected F1-Score**: 0.7-0.9 range

### Fraud Data Dataset
- **Class Distribution**: Moderately imbalanced (~10% fraud)
- **Expected Best Model**: Usually ensemble methods
- **Expected F1-Score**: 0.8-0.95 range

## Troubleshooting

### Common Issues

1. **Import Errors**: Install missing packages with `pip install -r requirements.txt`
2. **Memory Issues**: Reduce dataset size or use basic implementation
3. **Slow Training**: Use basic implementation or reduce hyperparameter search space
4. **XGBoost/LightGBM Not Available**: Basic implementation will work without these

### Performance Tips

1. **For Large Datasets**: Use LightGBM or reduce hyperparameter search
2. **For Quick Results**: Use basic implementation
3. **For Best Performance**: Use advanced implementation with all models

## Conclusion

This implementation provides a comprehensive solution for Task 2, covering:
- ✅ Data preparation for both datasets
- ✅ Train-test split with stratification
- ✅ Multiple model types (Logistic Regression + Ensemble)
- ✅ Appropriate metrics for imbalanced data
- ✅ Hyperparameter tuning
- ✅ Comprehensive evaluation and visualization
- ✅ Clear model comparison and justification

The solution is modular, well-documented, and provides both basic and advanced implementations to suit different needs and computational resources. 