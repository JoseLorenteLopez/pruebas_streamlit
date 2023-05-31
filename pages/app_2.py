import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Crear el título en Streamlit
#st.title('FORECAST SEGUIDORES REDES SOCIALES')

# Estilo CSS para el contenedor del título
titulo_css = """
    <style>
        .titulo-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }

        .titulo {
            font-size: 36px;
            margin: 0;
        }
    </style>
"""

# Título de la página
st.markdown(titulo_css, unsafe_allow_html=True)
st.markdown("<div class='titulo-container'><h1 class='titulo'>FORECAST SEGUIDORES REDES SOCIALES</h1><div>", unsafe_allow_html=True)

# Widget para cargar archivos
uploaded_file = st.sidebar.file_uploader("Por favor, suba su archivo CSV", type=['csv'])

# Espacio adicional
for _ in range(20):  # Puedes ajustar este número para obtener el espacio que desees
    st.sidebar.text("\n")

# Mostrar la imagen
#st.sidebar.image("logo_chiringuito.png", width=150) 

if uploaded_file is not None:
    # Carga de datos
    df = pd.read_csv(uploaded_file, parse_dates=['Month'], index_col='Month')

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

    # Mostrar el gráfico en Streamlit
    st.pyplot(plot())

    # Crear varios gráficos en Streamlit
    for _ in range(1):
        st.pyplot(plot())


