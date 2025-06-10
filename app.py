from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib  # atau bisa tetap pakai joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Tambahan untuk komunikasi dengan frontend

# Load model
model = joblib.load('model_regresi_padi.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Opsi 1: Menggunakan format array seperti contoh
    if 'features' in data:
        features = np.array(data['features']).reshape(1, -1)
    else:
        # Opsi 2: Tetap menggunakan format individual fields seperti kode asli Anda
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

if __name__ == '__main__':
    app.run(debug=True)