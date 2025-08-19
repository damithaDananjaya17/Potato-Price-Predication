from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load(r'D:\RESERCH\Vegitable price prediction\Model\potato_price_model.pkl')
le_veg, le_var, le_mkt, le_prov, le_sell = joblib.load(r'D:\RESERCH\Vegitable price prediction\Model\label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and clean input from form
        date = request.form['date'].strip()
        vegetable = request.form['vegetable'].strip()
        variety = request.form['variety'].strip()
        market = request.form['market'].strip()
        temperature = float(request.form['temperature'].strip())
        rainfall = float(request.form['rainfall'].strip())
        province = request.form['province'].strip()
        selling_market = request.form['selling_market'].strip()

        # Encode categorical features
        vegetable_enc = le_veg.transform([vegetable])[0]
        variety_enc = le_var.transform([variety])[0]
        market_enc = le_mkt.transform([market])[0]
        province_enc = le_prov.transform([province])[0]
        selling_market_enc = le_sell.transform([selling_market])[0]

        # Combine all features
        features = np.array([[temperature, rainfall, vegetable_enc, variety_enc, market_enc, province_enc, selling_market_enc]])

        prediction = model.predict(features)

        return render_template('index.html', prediction_text=f"Predicted Price: Rs. {prediction[0]:.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
