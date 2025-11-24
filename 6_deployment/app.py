"""Simple Flask API to serve predictions.

POST /predict
Content-Type: application/json
Body: either a single dict with feature names and values, or a list of dicts.

This service expects the input features to be unprocessed (original features).
It loads artifacts/preprocessor.joblib to transform incoming data, then loads artifacts/best_model.joblib.
"""
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)
MODEL_PATH = 'artifacts/best_model.joblib'
PREPROC_PATH = 'artifacts/preprocessor.joblib'

if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROC_PATH):
    raise RuntimeError('Run make prep and make train before starting the API')

model = joblib.load(MODEL_PATH)
preproc = joblib.load(PREPROC_PATH)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    else:
        df = pd.DataFrame(payload)
    # Ensure columns ordering: use columns saved earlier if available
    # For this example, expect the same columns as training data.
    X_pre = preproc.transform(df)
    preds = model.predict(X_pre).tolist()
    probs = model.predict_proba(X_pre)[:,1].tolist()
    return jsonify({'predictions': preds, 'probabilities': probs})

if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')