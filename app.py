from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import requests

app = Flask(__name__)


# GitHub URLs for models (replace with your actual raw URLs)
MODEL_URLS = {
    'models/best_regression_model.joblib': 'https://github.com/FSOPredict/be-ml-fsopredict/blob/main/models/best_regression_model.joblib',
    'models/best_regression_model_vis.joblib': 'https://github.com/FSOPredict/be-ml-fsopredict/blob/main/models/best_regression_model_vis.joblib',
    'models/best_regression_model_conditions.joblib': 'https://github.com/FSOPredict/be-ml-fsopredict/blob/main/models/best_regression_model_conditions.joblib',
}

# Download function
def download_model_if_missing(local_path, url):
    if not os.path.exists(local_path):
        print(f"Model not found at {local_path}. Downloading from {url}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded and saved: {local_path}")
        else:
            raise Exception(f"Failed to download model from {url} (Status code: {response.status_code})")

# Load models with auto-download
def load_model(path):
    if path in MODEL_URLS:
        download_model_if_missing(path, MODEL_URLS[path])
    return joblib.load(path)

# Load models
model_ber = load_model('models/best_regression_model.joblib')
model_vis = load_model('models/best_regression_model_vis.joblib')
model_conditions = load_model('models/best_regression_model_conditions.joblib')


# Mapping for weather condition code to label
conditions_mapping = {
    0: "Overcast, cloudy-day",
    1: "Overcast, cloudy-night",
    2: "Rain, Overcast-day",
    3: "Rain, Overcast-night",
    4: "Rain, Partially cloudy-day",
    5: "Rain, Partially cloudy-night",
    6: "clear-day",
    7: "clear-night",
    8: "fog",
    9: "partly-cloudy-day",
    10: "partly-cloudy-night"
}

# Visibility-dependent q
def get_q_from_visibility(V):
    if V > 50:
        return 1.6
    elif 6 < V <= 50:
        return 1.3
    elif 1 < V <= 6:
        return 0.16 * V + 0.34
    elif 0.5 < V <= 1:
        return V - 0.5
    else:
        return 0

# Attenuation calculation
def compute_attenuation(V, wavelength_nm=1550):
    q = get_q_from_visibility(V)
    A = (3.91 / V) * (wavelength_nm / 550) ** (-q)
    return A, q

# Feature engineering for visibility/condition models
def prepare_visibility_features(data):
    df = pd.DataFrame([data])
    df_vis = df[['temp', 'precip', 'humidity', 'windspeed', 'uvindex', 'dew', 'hour']].copy()
    df_vis['dew_uvindex_interaction'] = df_vis['dew'] * df_vis['uvindex']
    df_vis['dew_windspeed_interaction'] = df_vis['dew'] * df_vis['windspeed']
    df_vis['temp_humidity_interaction'] = df_vis['temp'] * df_vis['humidity']
    df_vis['humidity_squared'] = df_vis['humidity'] ** 2
    df_vis['humidity_cubed'] = df_vis['humidity'] ** 3
    df_vis['log_precip'] = np.log1p(df_vis['precip'])
    df_vis['log_temp'] = np.log1p(df_vis['temp'])
    return df_vis

@app.route('/predict', methods=['POST'])
def predict_all():
    try:
        data = request.get_json()
        df_vis = prepare_visibility_features(data)

        vis = float(model_vis.predict(df_vis)[0])
        cond_code = int(model_conditions.predict(df_vis)[0])
        cond_label = conditions_mapping.get(cond_code, "Unknown")

        attenuation, q = compute_attenuation(vis)
        df_ber = pd.DataFrame([{
            'visibility': vis,
            'q': q,
            'Attenuation': attenuation,
            'Power': data['power'],
            'Range': data['range']
        }])
        ber = float(model_ber.predict(df_ber)[0])

        return jsonify({
            'predicted_visibility': vis,
            'predicted_condition_code': cond_code,
            'predicted_condition_label': cond_label,
            'attenuation': attenuation,
            'q_value': q,
            'ber_prediction': ber
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/visibility', methods=['POST'])
def predict_visibility():
    try:
        data = request.get_json()
        df_vis = prepare_visibility_features(data)
        vis = float(model_vis.predict(df_vis)[0])
        return jsonify({'visibility': vis})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/conditions', methods=['POST'])
def predict_conditions():
    try:
        data = request.get_json()
        df_vis = prepare_visibility_features(data)
        cond_code = int(model_conditions.predict(df_vis)[0])
        cond_label = conditions_mapping.get(cond_code, "Unknown")
        return jsonify({
            'condition_code': cond_code,
            'condition_label': cond_label
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/ber', methods=['POST'])
def predict_ber():
    try:
        data = request.get_json()
        if 'visibility' not in data:
            return jsonify({'error': 'Missing visibility input for BER prediction'}), 400
        visibility = float(data['visibility'])
        q = get_q_from_visibility(visibility)
        attenuation, _ = compute_attenuation(visibility)

        df_ber = pd.DataFrame([{
            'visibility': visibility,
            'q': q,
            'Attenuation': attenuation,
            'Power': data['power'],
            'Range': data['range']
        }])
        ber = float(model_ber.predict(df_ber)[0])
        return jsonify({
            'visibility': visibility,
            'q': q,
            'attenuation': attenuation,
            'ber': ber
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
