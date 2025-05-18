from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Load model pipeline (tanpa 'mata_pelajaran')
with open('model_flag_v2.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'preflight'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        return response

    try:
        data = request.get_json()

        # Validasi input wajib
        required = ['jam_belajar', 'nilai_ujian_sebelumnya']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Field {field} diperlukan'}), 400

        # Tambah fitur nilai_ujian_rendah_flag sesuai nilai_ujian_sebelumnya
        nilai_ujian_sebelumnya = float(data.get('nilai_ujian_sebelumnya', 0))

        input_data = {
            'jam_belajar': float(data.get('jam_belajar', 0)),
            'nilai_ujian_sebelumnya': nilai_ujian_sebelumnya,
            'jam_tidur': float(data.get('jam_tidur', 6)),  # default: 6 jam
            'tingkat_ekonomi': int(data.get('tingkat_ekonomi', 3)),  # default: sedang
            'tingkat_motivasi': int(data.get('tingkat_motivasi', 3)),  # default: sedang
            'pertemanan': int(data.get('pertemanan', 3)),  # default: sedang
            'nilai_ujian_rendah_flag': 1 if nilai_ujian_sebelumnya <= 30 else 0
        }

        # Bentuk DataFrame untuk model
        df = pd.DataFrame([input_data])

        # Prediksi
        prediction = model.predict(df)[0]
        prediction = np.clip(prediction, 0, 100)

        # Response
        response = jsonify({
            'prediksi_nilai': round(float(prediction), 2),
            'status': 'success'
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
