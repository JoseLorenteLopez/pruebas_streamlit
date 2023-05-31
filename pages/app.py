import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Carga de datos
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

# Preparar datos para el pronóstico
df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)

# Dividir los datos
train_df, test_df = train_test_split(df, test_size=0.3, shuffle=False)

# Entrenar el modelo
model = ARIMA(train_df, order=(5,1,0))
model_fit = model.fit()

# Pronóstico
forecast = model_fit.forecast(steps=len(test_df))

def plot():
    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(10, 6))  # Ajusta el tamaño de la figura aquí

    # Trazar los datos de entrenamiento
    ax.plot(train_df, 'b', label='Train')

    # Trazar los datos de prueba
    ax.plot(test_df, 'g', label='Test')

    # Trazar el pronóstico
    ax.plot(test_df.index, forecast, 'r', label='Forecast')

    ax.legend()

    return fig

# Crear el título en Streamlit
st.title('FORECAST SEGUIDORES REDES SOCIALES')

# Mostrar el gráfico en Streamlit
st.pyplot(plot())

# Crear varios gráficos en Streamlit
for _ in range(8):
    st.pyplot(plot())
