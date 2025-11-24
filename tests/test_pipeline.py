import os
import subprocess
import sys
def test_prep_and_train():
    # run prep
    r = subprocess.run([sys.executable, '3_data_preparation/prep_pipeline.py'])
    assert r.returncode == 0
    # run train
    r2 = subprocess.run([sys.executable, '4_modeling/train_model.py'])
    assert r2.returncode == 0
    # check artifacts
    assert os.path.exists('artifacts/best_model.joblib')
    assert os.path.exists('artifacts/preprocessor.joblib')
