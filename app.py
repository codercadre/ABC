

import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('delivery_delay.sav')

st.title('Delivery Delay Prediction')
st.write('Enter the features below to predict if there will be a delivery delay.')

# Define the input features based on X.columns
feature_names = [
    'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
    'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
    'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

# Create input fields for each feature
input_data = {}
for feature in feature_names:
    # Using number_input for numerical features, you can customize type/range if needed
    input_data[feature] = st.number_input(f'Enter {feature.replace("_", " ")}:', value=0.0)

# Create a button to make predictions
if st.button('Predict Delivery Delay'):
    # Convert input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction Results:')
    if prediction[0] == 1:
        st.error('Prediction: There will be a Delivery Delay!')
    else:
        st.success('Prediction: No Delivery Delay Expected.')

    st.write(f'Probability of No Delay: {prediction_proba[0][0]:.2f}')
    st.write(f'Probability of Delay: {prediction_proba[0][1]:.2f}')


st.write('---')
st.write('Model Details: Logistic Regression')

