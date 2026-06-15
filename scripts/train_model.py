import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # Find the threshold where False Positive Rate == False Negative Rate
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'n_estimators': 150,
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    # Using modern early stopping callback
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

def main():
    print("🎙️ Training LightGBM Model with Optuna Tuning")
    print("Loading extracted features...")
    data_path = '/Users/hardik/Desktop/Mars/Deepfake_Audio/data/extracted_features.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Features file not found at {data_path}. Run extract_features.py first.")
        return
        
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1)
    y = df['label']
    
    # 80-20 Train-Test Split (Stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Further split train for validation in Optuna
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print("\nStarting Optuna Hyperparameter Optimization (15 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_tr, y_tr, X_val, y_val), n_trials=15)
    
    print("✅ Best hyperparameters found:", study.best_params)
    
    print("\nTraining final LightGBM model with best parameters...")
    best_params = study.best_params
    best_params['n_estimators'] = 300
    best_params['verbose'] = -1
    
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Evaluation
    print("\nEvaluating model on hold-out test set...")
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    eer, eer_threshold = compute_eer(y_test, y_scores)
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Per-class accuracy
    genuine_acc = cm[0,0] / (cm[0,0] + cm[0,1])
    fake_acc = cm[1,1] / (cm[1,0] + cm[1,1])
    
    print(f"\n🏆 RESULTS")
    print(f"Overall Accuracy: {acc*100:.2f}% (Requirement: >= 80%)")
    print(f"F1 Score:         {f1*100:.2f}% (Requirement: >= 80%)")
    print(f"Equal Error Rate: {eer*100:.2f}% (Requirement: <= 12%)")
    print(f"Genuine Accuracy: {genuine_acc*100:.2f}% (Requirement: >= 75%)")
    print(f"Deepfake Accuracy:{fake_acc*100:.2f}% (Requirement: >= 75%)")
    
    # Save Model
    os.makedirs('/Users/hardik/Desktop/Mars/Deepfake_Audio/models', exist_ok=True)
    model_path = '/Users/hardik/Desktop/Mars/Deepfake_Audio/models/lgbm_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    # Save Report
    report = f"""================ FINAL EVALUATION REPORT ================
Model Evaluated: LightGBM (Optuna Tuned)
Overall Accuracy: {acc*100:.2f}%
F1 Score:         {f1*100:.2f}%
Equal Error Rate: {eer*100:.2f}%
  EER Threshold:  {eer_threshold:.4f}

Per-Class Accuracy:
  - Genuine (Real) Speech: {genuine_acc*100:.2f}%
  - Deepfake (Fake) Speech: {fake_acc*100:.2f}%

Confusion Matrix:
[[{cm[0,0]} {cm[0,1]}]
 [{cm[1,0]} {cm[1,1]}]]
=========================================================
"""
    os.makedirs('/Users/hardik/Desktop/Mars/Deepfake_Audio/results', exist_ok=True)
    with open('/Users/hardik/Desktop/Mars/Deepfake_Audio/results/evaluation_report.txt', 'w') as f:
        f.write(report)
        
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Genuine', 'Deepfake'], yticklabels=['Genuine', 'Deepfake'])
    plt.title('LightGBM Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('/Users/hardik/Desktop/Mars/Deepfake_Audio/results/confusion_matrix.png')
    plt.close()

if __name__ == '__main__':
    main()
