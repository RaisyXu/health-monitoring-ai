# app.py
import streamlit as st
import joblib
import pandas as pd
import urllib.request
import os

# ----------------------
# Load your trained model
# ----------------------

MODEL_PATH = "random_forest_7d_model.pkl"
HF_URL = "https://huggingface.co/RaisyXu/property_7d/resolve/main/random_forest_7d_model.pkl"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(HF_URL, MODEL_PATH)

model = joblib.load(MODEL_PATH)

# ----------------------
# Expected Feature List
# ----------------------

expected_features = [
    'rent', 'property_size', 'photo_count', 'days_since_activation',
    'locality_freq', 'BHK_type', 'bathroom', 'floor', 'total_floor',
    'building_type_IF', 'building_type_IH', 'furnishing_SEMI_FURNISHED',
    'furnishing_NOT_FURNISHED', 'lease_type_FAMILY', 'lease_type_BACHELOR',
    'lease_type_COMPANY', 'parking_NONE', 'parking_TWO_WHEELER',
    'parking_FOUR_WHEELER', 'lift', 'gym', 'swimming_pool', 'pin_code',
    'latitude', 'longitude'
]

# ----------------------
# Streamlit UI
# ----------------------

st.title("🏡 Predict Property Interactions (7-Day)")

st.write("""
This app predicts the number of interactions (views, saves, inquiries) a property listing will receive over 7 days, based on features like rent, size, photos, and locality popularity.
""")

# Sidebar for user input
st.sidebar.header("Enter Property Features")

# Property features inputs
rent = st.sidebar.number_input('Monthly Rent (in $)', min_value=0, value=15000)
property_size = st.sidebar.number_input('Property Size (sqft)', min_value=0, value=800)
photo_count = st.sidebar.number_input('Number of Photos', min_value=0, value=5)
property_age_days = st.sidebar.number_input('Property Age (days)', min_value=0, value=30)
locality_freq = st.sidebar.number_input('Locality Popularity (Listing Count)', min_value=0, value=100)

# ----------------------
# Build the complete input features
# ----------------------

# Mapping user inputs
user_inputs = {
    'rent': rent,
    'property_size': property_size,
    'photo_count': photo_count,
    'days_since_activation': property_age_days,
    'locality_freq': locality_freq,
}

# Fill in default 0 for missing features
full_input_features = {feature: 0 for feature in expected_features}
full_input_features.update(user_inputs)

# Convert to DataFrame
input_df = pd.DataFrame([full_input_features])

# ----------------------
# Predict
# ----------------------

if st.button('Predict 7-Day Interactions'):
    prediction = model.predict(input_df)
    st.success(f"🎯 Predicted 7-Day Interactions: {prediction[0]:.2f}")
