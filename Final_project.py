import streamlit as st
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf  

# Apply Custom Background Style (Thick Blue)
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
    /* Sidebar Styling */
    .css-1d391kg, .css-1v3fvcr, .stSidebar {
        background-color: skyblue !important; /* Set sidebar background to sky blue */
        color: black !important; /* Ensure text is visible */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load models
models = {
    "Insurance Fraud Model": r"C:\Users\User\Downloads\insurance_fraud_model.pkl",
    "Fraud RF Model": r"C:\Users\User\Downloads\fraud_rf_model.pkl",
    "Fraud NN Model": r"C:\Users\User\Downloads\fraud_nn_model.pkl",
    "Customer Segmentation Model": r"C:\Users\User\Downloads\customer_segmentation_dataset.pkl"
}

# Define feature sets for each model
feature_sets = {
    "Insurance Fraud Model": ['Customer_Age', 'Gender', 'Policy_Type', 'Annual_Income', 'Claim_History', 'Premium_Amount', 'Claim_Amount', 'Risk_Score'],
    "Fraud RF Model": ['Claim_Amount', 'Claim_Type', 'Suspicious_Flags', 'Claim_to_Amount_Ratio'],
    "Fraud NN Model": ['Claim_Amount', 'Claim_Type', 'Suspicious_Flags', 'Claim_to_Amount_Ratio'],
    "Customer Segmentation Model": ['Age', 'Annual_Income', 'Policy_count', 'Total_Premium_paid', 'Claim_Frequency', 'Policy_Upgrades']
}

# Encode categorical values
encoding_maps = {
    "Gender": {"Male": 0, "Female": 1},
    "Policy_Type": {"Property": 0, "Auto": 1, "Health": 2, "life": 4},
    "Claim_Type": {"Accident": 0, "Fire": 1, "Theft": 2},
    "Suspicious_Flags": {"Suspicious": 1, "Normal": 0},
    "Risk_Score": {"Low": 1, "Medium": 2, "High": 3}
}

def load_model(model_path):
    """ Load model from pickle file """
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

# Load selected model only after selection
model_path = models[selected_model]
model = load_model(model_path)

st.title(f"{selected_model} Prediction")

# Get feature list based on selected model
features = feature_sets[selected_model]
user_inputs = {}

for feature in features:
    if feature in encoding_maps:  # If feature needs encoding
        user_inputs[feature] = st.selectbox(f"Enter {feature}", list(encoding_maps[feature].keys()))
    else:
        user_inputs[feature] = st.text_input(f"Enter {feature}")

if st.button("Predict"):
    try:
        # Convert inputs safely
        input_values = []
        for feature in features:
            value = user_inputs[feature].strip()  # Remove any extra spaces

            if feature in encoding_maps:
                input_values.append(encoding_maps[feature][value])  # Encode categorical values
            else:
                try:
                    input_values.append(float(value))  # Convert to float safely
                except ValueError:
                    st.error(f"Invalid input for {feature}. Please enter a valid number.")
                    st.stop()

        input_array = np.array(input_values).reshape(1, -1)
        prediction = model.predict(input_array)

        # Ensure prediction is a single value, not an array
        if isinstance(prediction, np.ndarray):
            prediction = prediction[0]
        elif isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        # Convert output labels for better readability
        result_map = {0: "This is 0, so it is Genuine Claims", 1: "This is 1, so it is Fraud Claims"}

        # Customer Segmentation Mapping
        cluster_map = {
            1: "1st Group",
            2: "2nd Group",
            3: "3rd Group",
            4: "4th Group"
        }

        if selected_model == "Customer Segmentation Model":
            prediction_text = f"Prediction: {cluster_map.get(int(prediction), f'Cluster {prediction}')} 🎯"
        elif selected_model in ["Fraud RF Model", "Fraud NN Model", "Insurance Fraud Model"]:
            prediction_text = f"Prediction: {result_map.get(int(prediction), prediction)} 🎯"
        else:
            prediction_text = f"Prediction: {prediction} 🎯"

        # 🔴 **Display prediction text in red**
        st.markdown(f'<p style="color:red; font-size:20px; font-weight:bold;">{prediction_text}</p>', unsafe_allow_html=True)

        # 🎈 Display balloons on prediction
        st.balloons()

    except Exception as e:
        st.error(f"Error in prediction: {e}")

