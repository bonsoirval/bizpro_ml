"""EDA script

Loads the sklearn breast cancer dataset, saves a raw CSV, computes summary statistics,
and writes a simple EDA report to artifacts/.
"""
import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

os.makedirs('artifacts', exist_ok=True)

data = load_breast_cancer(as_frame=True)
df = data.frame
# save raw
df.to_csv('artifacts/raw_data.csv', index=False)

# summary stats
summary = df.describe().T
summary.to_csv('artifacts/summary_stats.csv')

# class balance
class_counts = df['target'].value_counts().rename_axis('target').reset_index(name='counts')
class_counts.to_csv('artifacts/class_counts.csv', index=False)

# simple histogram for first three features
for col in df.columns[:3]:
    plt.figure(figsize=(6,3))
    df[col].hist(bins=30)
    plt.title(col)
    plt.tight_layout()
    plt.savefig(f'artifacts/hist_{col}.png')
    plt.close()

print('EDA complete. Artifacts saved to artifacts/*.')