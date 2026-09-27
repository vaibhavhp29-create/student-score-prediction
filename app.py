import streamlit as st
import pickle

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# UI
st.title("Student Score Predictor")

st.write("Enter the number of hours you study.")

hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=1.0
)

if st.button("Predict Score"):
    prediction = model.predict([[hours]])

    st.success(f"Predicted Score: {prediction[0]:.2f}")