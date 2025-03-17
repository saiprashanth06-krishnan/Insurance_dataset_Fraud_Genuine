import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Apply Custom Background Style (Thick Blue)
st.markdown(
    """
    <style>
    body {
        background-color: #004080; /* Dark Blue */
        color: white;
    }
    .stApp {
        background-color: #004080; /* Dark Blue */
    }
    h1, h2, h3, h4, h5, h6, label {
        color: yellow !important; /* Set main headings and labels to yellow */
        font-weight: bold;
    }
    .stButton>button {
        background-color: #FFA500; /* Orange button */
        color: black;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #FF4500; /* Darker Orange on Hover */
    }
    .css-1d391kg, .css-1v3fvcr, .stSidebar {
        background-color: skyblue !important; /* Set sidebar background to sky blue */
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load models
models = {
    "Insurance Fraud Model": r"C:\Users\User\Downloads\insurance_fraud_model.pkl",
    "Fraud Rf Model": r"C:\Users\User\Downloads\fraud_rf_model.pkl",
    "Customer Segmentation Model": r"C:\Users\User\Downloads\customer_segmentation_dataset.pkl"
}

# Define feature sets
feature_sets = {
    "Insurance Fraud Model": ['Customer_Age', 'Gender', 'Policy_Type', 'Annual_Income', 'Claim_History', 'Premium_Amount', 'Claim_Amount', 'Risk_Score'],
    "Fraud Rf Model": ['Claim_Amount', 'Claim_Type', 'Suspicious_Flags', 'Claim_to_Amount_Ratio'],
    "Customer Segmentation Model": ['Age', 'Annual_Income', 'Policy_count', 'Total_Premium_paid', 'Claim_Frequency', 'Policy_Upgrades']
}

# Encoding mappings
encoding_maps = {
    "Gender": {"Male": 0, "Female": 1},
    "Policy_Type": {"Property": 0, "Auto": 1, "Health": 2, "Life": 3},
    "Claim_Type": {"Accident": 0, "Fire": 1, "Theft": 2},
    "Risk_Score": {"Low": 1, "Medium": 2, "High": 3}
}

# Threshold limits
thresholds = {
    "Insurance Fraud Model": 0.22,
    "Fraud Rf Model": 0.70
}

# Customer Segmentation thresholds
def get_customer_segment(probability):
    if 0.00 <= probability < 0.30:
        return "0th Group"
    elif 0.30 <= probability < 0.50:
        return "1st Group"
    elif 0.50 <= probability < 0.80:
        return "2nd Group"
    else:
        return "3rd Group"

# Load model function
def load_model(model_path):
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Streamlit UI
st.sidebar.title("Select Model")
selected_model = st.sidebar.selectbox("Choose a model", list(models.keys()))
model_path = models[selected_model]
model = load_model(model_path)

st.title(f"{selected_model} Prediction")

features = feature_sets[selected_model]
user_inputs = {}

for feature in features:
    if feature in encoding_maps:
        user_inputs[feature] = st.selectbox(f"Enter {feature}", list(encoding_maps[feature].keys()))
    else:
        user_inputs[feature] = st.text_input(f"Enter {feature}")

if st.button("Predict"):
    try:
        input_values = []
        for feature in features:
            value = user_inputs[feature].strip()
            if feature in encoding_maps:
                input_values.append(encoding_maps[feature][value])
            else:
                try:
                    input_values.append(float(value))
                except ValueError:
                    st.error(f"Invalid input for {feature}. Please enter a valid number.")
                    st.stop()

        input_array = np.array(input_values).reshape(1, -1)
        prediction_proba = model.predict_proba(input_array)[:, 1][0]
        st.write(f"Fraud Probability: {prediction_proba:.4f}") 

        if selected_model == "Customer Segmentation Model":
            prediction_text = f"Prediction: {get_customer_segment(prediction_proba)} 🎯"
        else:
            threshold = thresholds[selected_model]
            prediction = int(prediction_proba > threshold)
            result_map = {0: "This is 0, so it is Genuine Claims", 1: "This is 1, so it is Fraud Claims"}
            prediction_text = f"Prediction: {result_map.get(prediction, prediction)} 🎯"

        st.markdown(f'<p style="color:red; font-size:20px; font-weight:bold;">{prediction_text}</p>', unsafe_allow_html=True)
        st.balloons()
    except Exception as e:
        st.error(f"Error in prediction: {e}")
