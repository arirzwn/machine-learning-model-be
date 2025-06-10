from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import logging

# Configure logging for Railway
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Konfigurasi CORS untuk memungkinkan akses dari frontend dan testing
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",
            "https://localhost:5173",
            "https://machine-learning-model-be-production.up.railway.app"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
}, supports_credentials=True)

# Load model
try:
    model_path = 'model_regresi_padi.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info("✅ Model loaded successfully")
    else:
        logger.error(f"❌ Model file not found at {model_path}")
        model = None
except Exception as e:
    logger.error(f"❌ Gagal load model: {str(e)}")
    model = None


@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'Rice Production Prediction API',
        'status': 'running',
        'model_loaded': model is not None,
        'endpoints': {
            'health': '/health',
            'predict': '/predict'
        }
    })


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Jawaban untuk preflight
        response = app.make_default_options_response()
        return response

    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

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
                data['pH']            ]])
        
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    logger.info(f"Starting Flask app on port {port}, debug={debug}")
    app.run(debug=debug, host='0.0.0.0', port=port)
