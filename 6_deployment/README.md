# 6. Deployment

This folder contains a simple Flask application to serve predictions.

- `app.py` expects artifacts/preprocessor.joblib and artifacts/best_model.joblib.
- Use a WSGI server (gunicorn) for production deployments and containerize with Docker.
