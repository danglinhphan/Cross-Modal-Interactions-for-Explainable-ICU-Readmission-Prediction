"""
Quick integration test script to validate backend endpoints /health, /features and /predict.
Usage:
  (1) Ensure backend is running (uvicorn Train_model.serve:app --port 8000)
  (2) Run this script:
     /Users/phandanglinh/Desktop/VRES/.venv/bin/python Train_model/quick_integration_test.py

It uses the configured NextJS frontend expected payload keys from `features.json`.
"""
import json
import os
import time
from pathlib import Path

try:
    import requests
except Exception:
    print('The `requests` library is required. Install with: pip install requests')
    raise

BASE_URL = os.environ.get('API_URL', 'http://localhost:8000')
FEATURES_PATH = Path('Train_model/outputs/readmission_tpe/features.json')


def load_features():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f'Features file not found at {FEATURES_PATH}')
    with FEATURES_PATH.open() as f:
        return json.load(f)


def build_payload(features):
    payload = {}
    for f in features:
        # Basic guesses: AGE -> 65; GENDER -> 'M'; numeric -> 0 or small sensible defaults
        if f == 'AGE':
            payload[f] = 65
        elif f == 'GENDER':
            payload[f] = 'M'
        elif f.endswith('_Avg'):
            payload[f] = 0.0
        elif f.endswith('_Std'):
            payload[f] = 0.0
        elif f.endswith('_Min'):
            payload[f] = 0.0
        elif f.endswith('_Max'):
            payload[f] = 0.0
        else:
            # fallback numeric placeholder
            payload[f] = 0
    return payload


def check_health():
    url = f'{BASE_URL}/health'
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    obj = r.json()
    print('health:', obj)
    return obj


def get_features():
    url = f'{BASE_URL}/features'
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    obj = r.json()
    print('features count:', obj.get('count'))
    return obj.get('features', [])


def post_predict(data):
    url = f'{BASE_URL}/predict'
    r = requests.post(url, json={'data': data}, timeout=20)
    r.raise_for_status()
    obj = r.json()
    print('predict response keys:', list(obj.keys()))
    return obj


if __name__ == '__main__':
    print(f'Testing API at {BASE_URL}')
    # Wait a short while for backend (if starting just now)
    time.sleep(1)

    # 1) Health
    try:
        health = check_health()
        if not health.get('model_loaded'):
            print('WARNING: model not loaded; ensure MODEL_PATH env var points to a .pkl and server log shows it loaded')
    except Exception as e:
        print('Health check failed:', str(e))
        raise

    # 2) Get features
    try:
        features = get_features()
        if not features:
            # fallback to reading the file if not served
            features = load_features()
            print('Loaded features from disk fallback; count:', len(features))
    except Exception as e:
        print('Failed to get features:', str(e))
        features = load_features()

    # 3) Build payload using feature list and call predict
    payload = build_payload(features)
    print('Posting predict with sample payload (first 10 keys):', list(payload.keys())[:10])

    try:
        res = post_predict(payload)
        if 'predictions' in res and len(res['predictions']) >= 1:
            print('Prediction OK - sample:', res['predictions'][0])
        else:
            print('Unexpected predict response:', res)
    except Exception as e:
        print('Predict failed:', str(e))
        raise

    print('Integration test finished successfully')
