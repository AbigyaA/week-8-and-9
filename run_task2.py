#!/usr/bin/env python3
"""
Task 2 Execution Script
Run this script to execute Task 2 - Model Building and Training

Usage:
    python run_task2.py [basic|advanced]

Options:
    basic    - Run basic implementation (Logistic Regression + Random Forest)
    advanced - Run advanced implementation (with XGBoost, LightGBM, hyperparameter tuning)
    
Default: basic
"""

import sys
import os

def main():
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = 'basic'
    
    print("="*60)
    print("TASK 2: MODEL BUILDING AND TRAINING")
    print("="*60)
    
    if mode == 'basic':
        print("Running BASIC implementation...")
        print("Models: Logistic Regression + Random Forest")
        print("Features: Standard evaluation metrics")
        
        try:
            from task2_model_building import main as run_basic
            run_basic()
        except ImportError as e:
            print(f"Error importing basic implementation: {e}")
            print("Make sure task2_model_building.py exists")
            return 1
    
    elif mode == 'advanced':
        print("Running ADVANCED implementation...")
        print("Models: Logistic Regression + Random Forest + Gradient Boosting + XGBoost + LightGBM")
        print("Features: Hyperparameter tuning + Comprehensive evaluation")
        
        try:
            from task2_advanced_models import main as run_advanced
            run_advanced()
        except ImportError as e:
            print(f"Error importing advanced implementation: {e}")
            print("Make sure task2_advanced_models.py exists")
            return 1
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: basic, advanced")
        return 1
    
    print("\n" + "="*60)
    print("Task 2 completed successfully!")
    print("="*60)
    return 0

if __name__ == "__main__":
    exit(main()) 