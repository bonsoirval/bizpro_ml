# 3. Data Preparation

This folder contains the preprocessing pipeline.

- `prep_pipeline.py` fits a SimpleImputer + StandardScaler (via ColumnTransformer) on training data
  and saves:
  - artifacts/preprocessor.joblib
  - artifacts/X_train.npy, X_val.npy, X_test.npy
  - artifacts/y_*.npy
  - artifacts/feature_columns.csv

Notes:
- Replace the data source with your raw data in production.
- For categorical columns, add an OneHotEncoder or similar transformer.
