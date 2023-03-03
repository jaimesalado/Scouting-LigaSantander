import streamlit as st
import pandas as pd
import joblib
import numpy as np
from gtts import gTTS
import io


style = f"""
<style>
    .appview-container .main .block-container{{
        max-width: 70%;
    }}
</style>
"""

st.markdown(style, unsafe_allow_html=True)


# imagen
from PIL import Image
image = ('laliga.png')
st.image(image, use_column_width=True)

# titulo

st.header("Descubriendo el talento a través del arte del análisis deportivo (ScoutLiga⚽📊)")
st.write("---")


tabs = ["Rendimiento", "Goles", "Goles encajados (Porteros)"]
tab = st.sidebar.selectbox("Selecciona una opción", tabs)
if tab == "Rendimiento":
    with st.container():
        st.subheader("Potencial de Rendimiento del Jugador en la temporada 📈 :")
        st.write("---")
        col1, col2 = st.columns(2)
# posicion
        with col1:
            st.subheader("Posición")
            posicion = st.selectbox("Selecciona Posición del jugador : ", ('Portero', 'Defensa', 'Mediocampista', 'Delantero'))
# minutosjugados
        with col2:
            st.subheader("Minutos Jugados")
            minutos_jugados = st.number_input (" Introduce minutos jugados durante la temporada : ", min_value=0)

        col1, col2 = st.columns(2)
# goles
        with col1:
            st.subheader("Goles")
            goles = st.number_input (" Introduce goles marcados durante la temporada : ", min_value=0)
# conversiongoles
        with col2:
            st.subheader("Conversion goles %")
            conversiongoles = st.number_input (" Introduce efectividad de cara a portería durante la temporada : ", min_value=0.0, max_value=100.0)

        col1, col2 = st.columns(2)
# regatesexito
        with col1:
            st.subheader("Regates Exito %")
            regatesexito = st.number_input (" Introduce efectividad regatenado durante la temporada : ", min_value=0.0, max_value=100.0)
# Perdida Posesion
        with col2:
            st.subheader("Perdida Posesion")
            perdidaposesion = st.number_input (" Introduce balones perdidos por el jugador durante la temporada : ", min_value=0)

        col1, col2 = st.columns(2)
# Asistencias
        with col1:
            st.subheader("Asistencias")
            asistencias = st.number_input (" Introduce Asistencias realizadas por el jugador durante la temporada : ", min_value=0)
# Pases Claves
        with col2:
            st.subheader("Pases Claves")
            pasesclaves = st.number_input (" Introduce pases claves realizados por el jugador durante la temporada : ", min_value=0)

        col1, col2 = st.columns(2)
# PasesPrecisos
        with col1:
            st.subheader("Pases Precisos %")
            pasesprecisos = st.number_input (" Introduce efectividad de pases realizados por jugador durante la temporada : ", min_value=0.0, max_value=100.0)
# Entradas
        with col2:
            st.subheader("Entradas")
            entradas = st.number_input (" Introduce cantidad de entradas realizadas por el jugador durante la temporada : ", min_value=0)

        col1, col2 = st.columns(2)
# Intercepciones
        with col1:
            st.subheader("Intercepciones")
            intercepciones = st.number_input (" Introduce balones interceptados por el jugador durante la temporada : ", min_value=0)
# DuelosGanado
        with col2:
            st.subheader("Total Duelos Ganado %")
            exitoduelosganados = st.number_input (" Introduce efectividad de duelos ganados por el jugador durante la temporada : ", min_value=0.0, max_value=100.0)


        col1, col2 = st.columns(2)
# Faltas
        with col1:
            st.subheader("Faltas")
            Faltas = st.number_input (" Introduce cantidad de faltas realizadas por el jugador durante la temporada : ", min_value=0)
# ParadasPartido%'
        with col2:
            st.subheader("Paradas Partido %")
            exitoparadas = st.number_input (" Introduce efectividad de Paradas por Partido de el jugador durante la temporada: ", min_value=0.0, max_value=100.0)

        col1, col2 = st.columns(2)
# golesencajados
        with col1:
            st.subheader("Goles Encajados")
            golesancajados = st.number_input("Introduce Goles Encajados por el jugador durante la temporada ", min_value=0)
        with col2:
# Definir función de síntesis de voz
            def hablartexto(text):
                with io.BytesIO() as buffer:
                    # Crear archivo de audio mp3 utilizando gTTS
                    tts = gTTS(text, lang='es')
                    tts.write_to_fp(buffer)
                    buffer.seek(0)
                    # Reproducir archivo de audio utilizando Streamlit
                    st.audio(buffer.read(), format="audio/mp3")
# boton
            st.subheader("Rendimiento")
            st.write("Pulsa botón para predecir rendimiento del jugador")
            if st.button("Predicción rendimiento"):
                # Unpickle classifier
                clf = joblib.load("ratings_model.pkl")
    
    # datos entradas
                laliga = pd.DataFrame([[posicion, minutos_jugados, goles, conversiongoles, regatesexito, perdidaposesion, asistencias, pasesclaves, pasesprecisos, entradas, intercepciones, exitoduelosganados, Faltas, exitoparadas, golesancajados       ]], 
                                      columns = ["Posicion", "Minutos", "Goles", "Conversion goles %", "Regates Exito %","Perdida Posesion", "Asistencias", "Pases Claves", "Pases Precisos %",
                                      "Entradas", "Intercepciones", "Total Duelos Ganado %", "Faltas",
                                      "Paradas Partido %", "Goles Encajados"])
                laliga['Posicion'].replace(['Portero', 'Defensa', 'Mediocampista', 'Delantero'], [1., 2., 3., 4.], inplace=True)

    
    # prediccion
                prediction = clf.predict(laliga)[0]
                redondeo = np.round(prediction, 2)

    
    # salida
                st.text(f"El rendimiento del jugador es:  {redondeo}")
                hablartexto(f"El rendimiento del jugador durante la temporada será: {redondeo}")

elif tab == "Goles":
    with st.container():
        st.subheader("Pronóstico goleador del Jugador 🎯")
        st.write("---")
        col1, col2 = st.columns(2)
# posiciongoleador
        with col1:
            st.subheader("Posición")
            posiciongoles = st.selectbox("Selecciona Posición del jugador : ", ('Portero', 'Defensa', 'Mediocampista', 'Delantero'))
# minutosjugadosgoleador
        with col2:
            st.subheader("Minutos Jugados")
            minutos_jugadosgoles = st.number_input (" Introduce minutos jugados durante la temporada : ", min_value=0)
# tirostotalesgoleador
        with col1:
            st.subheader("Tiros Totales")
            tirostotalesgoles = st.number_input (" Introduce tiros realizados por el jugador durante la temporada : ", min_value=0)
# conversiongolesgoleador
        with col2:
            st.subheader("Conversion goles %")
            conversiongolesgoles = st.number_input (" Introduce efectividad de cara a portería durante la temporada : ", min_value=0.0, max_value=100.0)
# botongoleador

            def hablartextogoles(text):
                with io.BytesIO() as buffer:
                    # Crear archivo de audio mp3 utilizando gTTS
                    tts = gTTS(text, lang='es')
                    tts.write_to_fp(buffer)
                    buffer.seek(0)
                    # Reproducir archivo de audio utilizando Streamlit
                    st.audio(buffer.read(), format="audio/mp3")
            st.subheader("Goles")
            st.write("Pulsa botón para predecir goles del jugador")
            if st.button("Predicción goles"):
                # Unpickle classifier
                clfgoles = joblib.load("goals_model.pkl")
    
    # datos entradas
                laliga = pd.DataFrame([[posiciongoles, minutos_jugadosgoles, tirostotalesgoles, conversiongolesgoles]], 
                                      columns = ["Posicion", "Minutos", "Tiros Totales", "Conversion goles %", ])
                laliga['Posicion'].replace(['Portero', 'Defensa', 'Mediocampista', 'Delantero'], [1., 2., 3., 4.], inplace=True)

    
    # prediccion
                predicciongoles = clfgoles.predict(laliga)[0]
                prediccion_entero = round(predicciongoles)


    
    # salida
                st.text(f"El Pronóstico goleador del jugador es: {prediccion_entero}")
                hablartextogoles(f"El Pronóstico goleador del jugador es: {prediccion_entero}")


elif tab == "Goles encajados (Porteros)":
    with st.container():
        st.subheader("Pronóstico de goles encajados por el portero 🚫")
        st.write("---")
        col1, col2 = st.columns(2)
# minutosportero
        with col1:
            st.subheader("Minutos jugados")
            minutosportero = st.number_input (" Introduce minutos jugados por el portero durante la temporada : ", min_value=0)
# paradaspartidoportero
        with col2:
            st.subheader("Paradas Partido %")
            paradaspartidoportero = st.number_input ("Introduce porcentaje de paradas por partido durante la temporada: ", min_value=0.0, max_value=100.0)

# imbatidoportero
        with col1:
            st.subheader("Imbatido")
            imbatidoportero = st.number_input ("Introduce veces con portería a 0 durante la temporada : ", min_value=0)
# botonportero
        with col2:
            st.subheader("Goles encajados: ")
            st.write("Pulsa botón para predecir goles encajados del portero")
           
            def hablartextogolesencajados(text):
                with io.BytesIO() as buffer:
                    # Crear archivo de audio mp3 utilizando gTTS
                    tts = gTTS(text, lang='es')
                    tts.write_to_fp(buffer)
                    buffer.seek(0)
                    # Reproducir archivo de audio utilizando Streamlit
                    st.audio(buffer.read(), format="audio/mp3")
            
           
            if st.button("Predicción goles encajados"):
                # Unpickle classifier
                clfgoles = joblib.load("goalsconceded_model.pkl")
    # datos entradas
                laliga = pd.DataFrame([[minutosportero,paradaspartidoportero,imbatidoportero]], 
                                      columns = ["Minutos", "Paradas Partido %", "Imbatido", ])
                

    
    # prediccion
                predicciongolesencajados = clfgoles.predict(laliga)[0]
                predicciongolesencajados_entero = round(predicciongolesencajados)


    
    # salida
                st.text(f"El Pronóstico de goles encajados es: {predicciongolesencajados_entero}")
                hablartextogolesencajados(f"El Pronóstico de goles encajados por el portero durante la temporada es : {predicciongolesencajados_entero}")

