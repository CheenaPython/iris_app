import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# App title
st.title("Iris Flower Species Prediction")
st.write("This app predicts the species of Iris flower using a trained Random Forest model.")

# Sidebar for input features
st.sidebar.header("Input Features")

# Input sliders
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    species = prediction[0]
    
    st.success(f"The predicted Iris species is: **{species}**")

