from pickle import load
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

model = load(open("../models/random_forest_classifier_default_42.sav", "rb"))
class_dict = {
    "0": "Diabetes Free",
    "1": "Strong Diabetes",
}


df = pd.read_csv("../data/raw/data.csv") 


num_variables = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']
scaler = StandardScaler()
scaler.fit(df[num_variables])

st.title("Diabetes")

val1 = st.slider("Pregnancies", min_value=0.0, max_value=15.0, step=0.1)
val2 = st.slider("Glucose", min_value=44.0, max_value=198.0, step=0.1)
val3 = st.slider("BloodPressure", min_value=44.0, max_value=122.0, step=0.1)
val4 = st.slider("BMI", min_value=0.0, max_value=100.0, step=0.1)
val5 = st.slider("DiabetesPedigreeFunction", min_value=0.078, max_value=1.781, step=0.001)
val6 = st.slider("Age", min_value=18.0, max_value=99.0, step=0.1)

if st.button("Predict"):
    data = np.array([[val1, val2, val3, val4, val5, val6]])
    data_normalized = scaler.transform(data)
    
    prediction = str(model.predict(data_normalized)[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)