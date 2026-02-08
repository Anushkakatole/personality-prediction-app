import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load model & scaler
# -----------------------------
with open("personality_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Personality Prediction", layout="centered")

st.title("🧠 Personality Type Prediction")
st.write("Adjust the sliders and predict the personality type")

st.divider()

# -----------------------------
# Feature names (MUST match training order)
# -----------------------------
feature_names = [
    'social_energy', 'alone_time_preference', 'talkativeness',
    'deep_reflection', 'group_comfort', 'party_liking',
    'listening_skill', 'empathy', 'organization',
    'leadership', 'risk_taking', 'public_speaking_comfort',
    'curiosity', 'routine_preference', 'excitement_seeking',
    'friendliness', 'planning', 'spontaneity',
    'adventurousness', 'reading_habit', 'sports_interest',
    'online_social_usage', 'travel_desire', 'gadget_usage',
    'work_style_collaborative', 'decision_speed'
]

# Personality label mapping
personality_map = {
    0: "Introvert",
    1: "Ambivert",
    2: "Extrovert"
}

# -----------------------------
# Input sliders
# -----------------------------
user_input = {}

for feature in feature_names:
    user_input[feature] = st.slider(
        label=feature.replace("_", " ").title(),
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

# Convert input to DataFrame (important for scaler)
input_df = pd.DataFrame([user_input])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Personality"):

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict class & probabilities
    predicted_class = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]

    # Convert class to text
    personality_label = personality_map.get(predicted_class, "Unknown")

    # -----------------------------
    # Display result
    # -----------------------------
    st.success(f"### 🧠 Predicted Personality Type: **{personality_label}**")

    # Confidence text
    confidence = np.max(probabilities) * 100
    st.write(f"**Prediction Confidence:** {confidence:.2f}%")

    # -----------------------------
    # Probability bar chart
    # -----------------------------
    prob_df = pd.DataFrame({
        "Personality Type": [personality_map[c] for c in model.classes_],
        "Probability": probabilities
    })

    st.subheader("Prediction Confidence Distribution")
    st.bar_chart(prob_df.set_index("Personality Type"))

    st.divider()
    st.caption("Model: Logistic Regression | Scaled Inputs Applied")
