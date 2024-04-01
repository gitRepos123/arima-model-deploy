import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import joblib

st.header('Electrical Production Based on ARIMA Model')

def make_response(days: int = 1) -> None:
    st.subheader(f"Prediction for next { days } days")
    days_array = [ x+1 for x in range(days) ]
    model = joblib.load('trained_model.pk1')
    predictions = model.predict(start = 398, end = 397 + days)
    data = pd.DataFrame({
        'Days': days_array,
        'Predictions': predictions
    })
    st.table(data)
    st.subheader("Days vs. Production Trend")
    fig = plt.figure()
    plt.xlabel('Days')
    plt.ylabel('Production')
    plt.plot(days_array, predictions, color = "green")
    st.pyplot(fig)

def init_app():
    days = st.slider(label = 'Input the number of days', min_value = 1, max_value = 7, step = 1)
    make_response(days)

init_app()
