import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load trained PyTorch model and scaler
model = torch.load("life_expectancy_model.pt", map_location=torch.device("cpu"))
model.eval()
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🌍 Life Expectancy Predictor")
st.markdown("Predict life expectancy using GDP, population, year, and continent.")

# User Input
year = st.number_input("Year", min_value=1950, max_value=2025, value=2007)
gdp_percap = st.number_input("GDP per Capita", value=5000.0)
population = st.number_input("Population", value=1_000_000)

continent = st.selectbox("Continent", ["Africa", "Americas", "Asia", "Europe", "Oceania"])

# One-hot encoding for continent
continent_map = {
    "Africa": [1, 0, 0, 0, 0],
    "Americas": [0, 1, 0, 0, 0],
    "Asia": [0, 0, 1, 0, 0],
    "Europe": [0, 0, 0, 1, 0],
    "Oceania": [0, 0, 0, 0, 1],
}

# Prepare input
user_input = [year, gdp_percap, population] + continent_map[continent]
user_input_np = np.array(user_input).reshape(1, -1)

# Scale input
scaled_input = scaler.transform(user_input_np)
input_tensor = torch.tensor(scaled_input, dtype=torch.float32)

# Predict
if st.button("Predict Life Expectancy"):
    with torch.no_grad():
        prediction = model(input_tensor).item()
    st.success(f"✅ Predicted Life Expectancy: **{prediction:.2f} years**")

# Footer
st.markdown("---")
st.markdown("Model trained on Gapminder dataset. For academic use only.")
