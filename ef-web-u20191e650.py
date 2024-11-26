import streamlit as st
import pickle
import pandas as pd

# Cargar los modelos guardados
with open('optimized_rf.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

with open('optimized_lr.pkl', 'rb') as lr_file:
    lr_model = pickle.load(lr_file)

# Interfaz de usuario en Streamlit
def main():
    st.title('Clasificación de Supervivencia en Titanic')

    st.sidebar.header('Parámetros de entrada')

    def user_input_parameters():
        Pclass = st.sidebar.selectbox('Clase de pasajero', [1, 2, 3])
        Sex = st.sidebar.selectbox('Sexo', ['Male', 'Female'])
        Age = st.sidebar.slider('Edad', 0, 80, 30)
        SibSp = st.sidebar.slider('Número de hermanos/esposos a bordo', 0, 8, 0)
        Parch = st.sidebar.slider('Número de padres/hijos a bordo', 0, 6, 0)
        Fare = st.sidebar.slider('Tarifa', 0.0, 500.0, 50.0)
        Embarked = st.sidebar.selectbox('Puerto de embarque', ['C', 'Q', 'S'])
        
        # Convertir los parámetros a un DataFrame
        data = {
            'Pclass': [Pclass],
            'Sex': [1 if Sex == 'Male' else 0],  # Codificar sexo
            'Age': [Age],
            'SibSp': [SibSp],
            'Parch': [Parch],
            'Fare': [Fare],
            'Embarked': [0 if Embarked == 'C' else 1 if Embarked == 'Q' else 2]  # Codificar embarque
        }
        features = pd.DataFrame(data)
        return features

    df = user_input_parameters()

    # Selección de modelo
    option = ['Random Forest', 'Logistic Regression']
    model = st.sidebar.selectbox('¿Qué modelo te gustaría usar?', option)

    st.subheader('Parámetros de entrada del usuario')
    st.write(df)

    # Predicción
    if st.button('Clasificar'):
        if model == 'Random Forest':
            prediction = rf_model.predict(df)
            result = 'Sobrevivió' if prediction[0] == 1 else 'No sobrevivió'
            st.success(f'Predicción: {result}')
        elif model == 'Logistic Regression':
            prediction = lr_model.predict(df)
            result = 'Sobrevivió' if prediction[0] == 1 else 'No sobrevivió'
            st.success(f'Predicción: {result}')

if __name__ == '__main__':
    main()
