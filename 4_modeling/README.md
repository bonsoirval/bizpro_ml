# 4. Modeling

This folder contains the training script that:
- Loads preprocessed arrays from artifacts/
- Runs GridSearchCV to tune a RandomForest
- Logs experiment to MLflow (local file store at ./mlruns/)
- Saves the best model to artifacts/best_model.joblib

To view MLflow UI:
```
mlflow ui --backend-store-uri file://$(pwd)/mlruns --default-artifact-root file://$(pwd)/mlruns
# then open http://127.0.0.1:5000 in your browser
```
