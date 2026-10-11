import pickle
import pandas as pd
import streamlit as st

st.title("Heart Disease Prediction System")
st.write("Enter the 4 patient details below.")

@st.cache_resource
def load_model():
    with open("svc_heart_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

age = st.number_input("Age (years)", min_value=1, max_value=120, value=55)
sex = st.selectbox("Sex", options=[1, 0],
                   format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest pain type (1 to 4)", options=[1, 2, 3, 4], index=3)
thalach = st.number_input("Max heart rate achieved", min_value=60,
                          max_value=220, value=130)

if st.button("Predict"):
    user_input = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "thalach": [thalach],
    })
    result = model.predict(user_input)[0]
    if result == 1:
        st.error("HEART DISEASE DETECTED")
    else:
        st.success("NO HEART DISEASE")
        