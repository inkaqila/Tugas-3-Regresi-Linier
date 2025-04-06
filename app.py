from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load model
model = joblib.load('model_regresi.pkl')

# Konstanta untuk kurs
USD_TO_IDR = 16000  # 1 USD = 16.000 IDR

# Fungsi prediksi
def predict_income(inputs):
    inputs_usd = np.array(inputs) / USD_TO_IDR
    prediction_usd = model.predict([inputs_usd])[0]
    prediction_idr = prediction_usd * USD_TO_IDR
    return prediction_idr

# Fungsi prediksi untuk CSV
def predict_from_csv(df):
    df_usd = df[['TV', 'Radio', 'Newspaper']] / USD_TO_IDR
    prediction_usd = model.predict(df_usd)
    prediction_idr = prediction_usd * USD_TO_IDR
    df['Prediksi Penjualan (Rp)'] = prediction_idr
    return df

# Fungsi buat grafik
def generate_plot(df, predictions):
    plt.figure(figsize=(8, 5))
    plt.scatter(df['TV'], df['Sales'], color='blue', label='Data Aktual')
    plt.scatter(df['TV'], predictions, color='red', label='Prediksi Model')
    plt.xlabel('TV Advertising Budget (USD)')
    plt.ylabel('Sales (USD)')
    plt.legend()
    plt.title('Grafik Regresi Linear - TV Budget vs Sales')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle form prediksi manual
@app.route('/predict', methods=['POST'])
def predict():
    try:
        tv = float(request.form['tv'])
        radio = float(request.form['radio'])
        newspaper = float(request.form['newspaper'])

        inputs = [tv, radio, newspaper]
        prediction = predict_income(inputs)

        # Baca dataset buat referensi grafik
        df = pd.read_csv('dataset/advertising.csv')
        predictions = model.predict(df[['TV', 'Radio', 'Newspaper']])
        plot_url = generate_plot(df, predictions)

        return render_template('index.html',
                               prediction=format_rupiah(prediction),
                               tips="Pertahankan konsistensi iklan di berbagai media untuk hasil yang lebih optimal.",
                               plot_url=plot_url)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Handle upload CSV
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        df_result = predict_from_csv(df.copy())
        df_result['Prediksi Penjualan (Rp)'] = df_result['Prediksi Penjualan (Rp)'].apply(format_rupiah)

        # Buat grafik
        predictions = model.predict(df[['TV', 'Radio', 'Newspaper']])
        plot_url = generate_plot(df, predictions)

        result = df_result.to_dict(orient='records')

        # Simpan hasil ke file CSV sementara untuk download
        df_result.to_csv('hasil_prediksi.csv', index=False)

        return render_template('index.html',
                               csv_results=result,
                               plot_url=plot_url,
                               download_link='/download')

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Endpoint untuk download hasil CSV
@app.route('/download')
def download_file():
    return send_file('hasil_prediksi.csv', as_attachment=True)

# Format rupiah
def format_rupiah(amount):
    return f"Rp {amount:,.0f}".replace(",", ".")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)

