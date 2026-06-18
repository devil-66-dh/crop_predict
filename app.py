import streamlit as st
import pandas as pd
import joblib

model = joblib.load("crop_model.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-image:
    linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.55)),
    url("https://images.unsplash.com/photo-1523741543316-beb7fc7023d8?auto=format&fit=crop&w=1600&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

header {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

.hero {
    text-align: center;
    margin-bottom: 40px;
}

.hero-title {
    font-size: 52px;
    font-weight: 900;
    color: white;
    text-shadow: 2px 4px 14px rgba(0,0,0,0.9);
}

.hero-subtitle {
    color: #F1FFF4;
    font-size: 19px;
    margin-top: 8px;
    text-shadow: 1px 2px 8px rgba(0,0,0,0.9);
}

.input-title {
    color: white;
    font-size: 36px;
    font-weight: 900;
    text-align: center;
    margin-bottom: 35px;
    text-shadow: 2px 4px 14px rgba(0,0,0,0.9);
}

label {
    color: white !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    text-shadow: 1px 2px 8px rgba(0,0,0,1);
}

/* Remove outer boxes */
[data-testid="stNumberInput"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Input field only */
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.90) !important;
    color: #102A22 !important;
    font-weight: 800 !important;
    border-radius: 14px !important;
    border: none !important;
    height: 48px !important;
}

/* Plus minus area */
[data-testid="stNumberInput"] button {
    background: rgba(15,23,42,0.85) !important;
    color: white !important;
    border: none !important;
}

.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 16px;
    border: none;
    background: linear-gradient(90deg, #1B4332, #40916C);
    color: white;
    font-size: 20px;
    font-weight: 900;
    margin-top: 25px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.35);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #081C15, #2D6A4F);
    color: white;
}

.result-card {
    margin-top: 35px;
    text-align: center;
}

.result-title {
    color: white;
    font-size: 25px;
    font-weight: 700;
    text-shadow: 2px 3px 10px rgba(0,0,0,1);
}

.crop-name {
    color: white;
    font-size: 64px;
    font-weight: 900;
    margin-top: 12px;
    text-shadow: 2px 5px 18px rgba(0,0,0,1);
}

.footer {
    text-align: center;
    color: white;
    margin-top: 30px;
    font-size: 14px;
    text-shadow: 1px 2px 8px rgba(0,0,0,1);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-title">🌾 Crop Recommendation System</div>
    <div class="hero-subtitle">AI Powered Smart Agriculture Assistant</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="input-title">
🌱 Enter Soil and Weather Details
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0, value=90)
    temperature = st.number_input("Temperature (°C)", value=25.00, format="%.2f")

with col2:
    P = st.number_input("Phosphorus (P)", min_value=0, value=42)
    humidity = st.number_input("Humidity (%)", value=80.00, format="%.2f")

with col3:
    K = st.number_input("Potassium (K)", min_value=0, value=43)
    rainfall = st.number_input("Rainfall (mm)", value=200.00, format="%.2f")

ph = st.number_input("Soil pH", value=6.50, format="%.2f")

predict = st.button("🌾 Recommend Crop")

if predict:
    sample = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    )

    prediction = model.predict(sample)
    crop = le.inverse_transform(prediction)[0]

    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">Recommended Crop</div>
        <div class="crop-name">🌱 {crop.title()}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
Built with Streamlit and Machine Learning
</div>
""", unsafe_allow_html=True)