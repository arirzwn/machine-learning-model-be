from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)


# Konfigurasi eksplisit agar preflight (OPTIONS) request juga diizinkan
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}}, supports_credentials=True)

# Load model
model = joblib.load('model_regresi_padi.pkl')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Jawaban untuk preflight
        response = app.make_default_options_response()
        return response

    data = request.json

    if 'features' in data:
        features = np.array(data['features']).reshape(1, -1)
    else:
        features = np.array([[ 
            data['Tahun'],
            data['Bulan'],
            data['Luas_Lahan'],
            data['Luas_Panen_ha'],
            data['Curah_Hujan_mm'],
            data['Suhu_Tanah'],
            data['Kelembapan'],
            data['pH']
        ]])
    
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})



# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))
#     app.run(debug=False, host='0.0.0.0', port=port)
