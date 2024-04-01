import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import joblib

st.header('Electrical Production Based ARIMA Model')

def make_response(days: int = 1) -> None:
    st.subheader(f"Prediction for next { days } days")
    days_array = [ x+1 for x in range(days) ]
    model = joblib.load('trained_model.pk1')
    predictions = model.predict(start = 1, end = days)
    data = pd.DataFrame({
        'Days': days_array,
        'Predictions': predictions
    })
    st.table(data)
    plt.xlabel('Days')
    plt.ylabel('Production')
    plt.plot(days_array, predictions)
    plt.show()

def init_app():
    days = st.slider(label = 'Input the number of days', min_value = 1, max_value = 7, step = 1)
    make_response(days)

init_app()
