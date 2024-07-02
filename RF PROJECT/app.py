from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("banking_crisis_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data, index=[0])

    # Apply Min-Max scaling to 'exch_usd' and 'inflation_annual_cpi' columns
    df[['exch_usd', 'inflation_annual_cpi']] = scaler.transform(df[['exch_usd', 'inflation_annual_cpi']])

    # Make prediction
    prediction = model.predict(df)

    # Decode prediction to original labels
    result = "YES , THERE WILL BE CRISIS" if prediction[0] == 1 else "NO , THERE WILL NOT BE ANY CRISIS"

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
