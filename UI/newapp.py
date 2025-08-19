import streamlit as st
import joblib
import numpy as np

# Load trained models
models = {
    "Welimada": joblib.load("welimada_model.pkl"),
    "Bandarawela": joblib.load("bandarawela_model.pkl"),
    "Nuwara Eliya": joblib.load("nuwaraeliya_model.pkl")
}

# Load corresponding encoders
encoders = {
    "Welimada": {
        "vegetable": joblib.load("welimada_vegetable_encoder.pkl"),
        "variety": joblib.load("welimada_variety_encoder.pkl"),
        "province": joblib.load("welimada_province_encoder.pkl"),
        "market": joblib.load("welimada_market_encoder.pkl"),
    },
    "Bandarawela": {
        "vegetable": joblib.load("bandarawela_vegetable_encoder.pkl"),
        "variety": joblib.load("bandarawela_variety_encoder.pkl"),
        "province": joblib.load("bandarawela_province_encoder.pkl"),
        "market": joblib.load("bandarawela_market_encoder.pkl"),
    },
    "Nuwara Eliya": {
        "vegetable": joblib.load("nuwaraeliya_vegetable_encoder.pkl"),
        "variety": joblib.load("nuwaraeliya_variety_encoder.pkl"),
        "province": joblib.load("nuwaraeliya_province_encoder.pkl"),
        "market": joblib.load("nuwaraeliya_market_encoder.pkl"),
    }
}

# Streamlit UI
st.title("Potato Price Prediction Tool 🥔")
st.markdown("Predict future potato selling prices based on market, weather and crop details.")

# Input form
selling_market = st.selectbox("Select Selling Market", ["Welimada", "Bandarawela", "Nuwara Eliya"])
vegetable = st.text_input("Vegetable", "Potato")
variety = st.text_input("Variety", "Granola")
province = st.text_input("Province", "Uva Province")

# Market-specific inputs
if selling_market == "Welimada":
    temperature = st.number_input("Welimada Temperature (°C)", value=25.0)
    rainfall = st.number_input("Welimada Rainfall (mm)", value=5.0)
elif selling_market == "Bandarawela":
    temperature = st.number_input("Bandarawela Temperature (°C)", value=22.0)
    rainfall = st.number_input("Bandarawela Rainfall (mm)", value=8.0)
else:
    temperature = st.number_input("Nuwara Eliya Temperature (°C)", value=20.0)
    rainfall = st.number_input("Nuwara Eliya Rainfall (mm)", value=10.0)

# Predict button
if st.button("Predict Price"):
    try:
        # Get correct model & encoders
        model = models[selling_market]
        encoder = encoders[selling_market]

        # Encode features using the market-specific label encoders
        veg_code = encoder["vegetable"].transform([vegetable])[0]
        var_code = encoder["variety"].transform([variety])[0]
        prov_code = encoder["province"].transform([province])[0]
        market_code = encoder["market"].transform([selling_market])[0]

        # Format input features
        input_features = np.array([[veg_code, var_code, temperature, rainfall, prov_code, market_code]])

        # Predict and display result
        prediction = model.predict(input_features)[0]
        st.success(f"Predicted Price at **{selling_market}**: Rs. {round(prediction, 2)}")
    
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Ensure that inputs like 'Vegetable', 'Variety', and 'Province' match training data.")
