"""Data preparation pipeline.

- Loads raw data (from sklearn for this example)
- Splits into train/val/test
- Fits a preprocessing pipeline (imputer + scaler)
- Saves processed datasets and the preprocessor artifact
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

os.makedirs('artifacts', exist_ok=True)

data = load_breast_cancer(as_frame=True)
X = data.frame.drop(columns=['target'])
y = data.frame['target']

# train/val/test split: 70/15/15
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.17647058823529413, # 0.15/0.85 ~ 0.17647
    random_state=42, stratify=y_trainval)

# simple numeric preprocessing
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
])

# fit preprocessor on train
preprocessor.fit(X_train)

# transform and save processed datasets
X_train_p = preprocessor.transform(X_train)
X_val_p = preprocessor.transform(X_val)
X_test_p = preprocessor.transform(X_test)

# Save as numpy for simplicity
import numpy as np
np.save('artifacts/X_train.npy', X_train_p)
np.save('artifacts/X_val.npy', X_val_p)
np.save('artifacts/X_test.npy', X_test_p)
y_train.to_numpy().tofile('artifacts/y_train.npy')
y_val.to_numpy().tofile('artifacts/y_val.npy')
y_test.to_numpy().tofile('artifacts/y_test.npy')

# Save column names and preprocessor
joblib.dump(preprocessor, 'artifacts/preprocessor.joblib')
pd.Series(numeric_cols).to_csv('artifacts/feature_columns.csv', index=False, header=False)

print('Data prep complete. Processed arrays and preprocessor saved to artifacts/.')