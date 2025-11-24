"""Evaluation script.

- Loads model and test data
- Computes and prints metrics
- Saves an evaluation report to artifacts/evaluation_report.json
"""
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
import json
import os

os.makedirs('artifacts', exist_ok=True)

model = joblib.load('artifacts/best_model.joblib')
X_test = np.load('artifacts/X_test.npy')
y_test = np.fromfile('artifacts/y_test.npy', dtype=float)

proba = model.predict_proba(X_test)[:,1]
pred = model.predict(X_test)
metrics = {
    'roc_auc': float(roc_auc_score(y_test, proba)),
    'accuracy': float(accuracy_score(y_test, pred)),
    'recall': float(recall_score(y_test, pred)),
    'precision': float(precision_score(y_test, pred))
}
print('Evaluation metrics:', metrics)
with open('artifacts/evaluation_report.json','w') as f:
    json.dump(metrics, f, indent=2)
print('Saved evaluation report to artifacts/evaluation_report.json')