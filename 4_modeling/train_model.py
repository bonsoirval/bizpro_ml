"""Train model script.

- Loads preprocessed numpy arrays from artifacts/
- Trains a RandomForest with GridSearchCV
- Logs parameters, metrics, and model artifact to MLflow (local file store)
- Saves best model to artifacts/best_model.joblib
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
import mlflow
import mlflow.sklearn

os.makedirs('artifacts', exist_ok=True)
# MLflow tracking URI (local)
mlflow.set_tracking_uri('file://' + os.path.abspath('mlruns'))
mlflow.set_experiment('crispdm_breast_cancer')

# load preprocessed arrays
X_train = np.load('artifacts/X_train.npy')
X_val = np.load('artifacts/X_val.npy')
X_test = np.load('artifacts/X_test.npy')
y_train = np.fromfile('artifacts/y_train.npy', dtype=float)
y_val = np.fromfile('artifacts/y_val.npy', dtype=float)
y_test = np.fromfile('artifacts/y_test.npy', dtype=float)

# combine train+val for GridSearch
X = np.vstack([X_train, X_val])
y = np.concatenate([y_train, y_val])

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 6, 12],
    'min_samples_split': [2, 5]
}

rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
gs = GridSearchCV(rfc, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)

with mlflow.start_run():
    gs.fit(X, y)
    best = gs.best_estimator_
    # evaluate on test
    proba = best.predict_proba(X_test)[:,1]
    pred = best.predict(X_test)
    metrics = {
        'roc_auc': float(roc_auc_score(y_test, proba)),
        'accuracy': float(accuracy_score(y_test, pred)),
        'recall': float(recall_score(y_test, pred)),
        'precision': float(precision_score(y_test, pred))
    }
    # log params and metrics
    mlflow.log_params(gs.best_params_)
    mlflow.log_metrics(metrics)
    # log model
    mlflow.sklearn.log_model(best, artifact_path='model')
    # also save locally
    joblib.dump(best, 'artifacts/best_model.joblib')
    print('Training complete. Metrics:', metrics)
    print('Best params:', gs.best_params_)