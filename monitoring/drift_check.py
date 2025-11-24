"""Monitoring sketch for drift detection.

- Loads reference data (train) and compares incoming batch using KS-tests per feature.
- Alerts by printing; replace with real alerting in production.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import os

def check_ks(ref, new, alpha=0.01):
    stat, p = ks_2samp(ref, new)
    return stat, p

if __name__ == '__main__':
    # load reference (train) data
    if not os.path.exists('artifacts/X_train.npy'):
        print('No artifacts found. Run make prep first.')
        exit(1)
    X_ref = np.load('artifacts/X_train.npy')
    # Simulate new batch by sampling from test
    X_new = np.load('artifacts/X_test.npy')
    # For each column (array columns)
    alerts = []
    for i in range(X_ref.shape[1]):
        stat, p = check_ks(X_ref[:,i], X_new[:,i])
        if p < 0.01:
            alerts.append({'feature_index': i, 'stat': float(stat), 'pvalue': float(p)})
    if alerts:
        print('Drift alerts:', alerts)
    else:
        print('No significant drift detected (alpha=0.01)')