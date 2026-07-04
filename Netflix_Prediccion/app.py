import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path

df = pd.read_csv("Base_De_Datos_Netflix.csv")

# ==========================
# Obtener listas para los SelectBox
# ==========================

generos = sorted(df["gender"].dropna().unique())
paises = sorted(df["country"].dropna().unique())
estados = sorted(df["state_province"].dropna().unique())
dispositivos = sorted(df["device_type"].dropna().unique())
calidades = sorted(df["quality"].dropna().unique())
ratings = sorted(df["user_rating"].dropna().unique())
tipos_contenido = sorted(df["content_type"].dropna().unique())
generos_contenido = sorted(df["genre_primary"].dropna().unique())
idiomas = sorted(df["language"].dropna().unique())
paises_origen = sorted(df["country_of_origin"].dropna().unique())
acciones = sorted(df["action"].dropna().unique())



# ==========================
# Configuración
# ==========================
ruta_logo = Path(__file__).parent / "Logonetflix.png"
st.image(ruta_logo, width=280)

st.set_page_config(
    page_title="Netflix Subscription Predictor",
    page_icon="🎬",
    layout="wide"
)



def cargar_css():
    ruta_css = Path(__file__).parent / "style.css"
    with open(ruta_css, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
cargar_css()
# ==========================
# Cargar modelo
# ==========================

modelo = joblib.load("modelo_random_forest.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")
columnas_modelo = joblib.load("columnas_modelo.pkl")

# ==========================
# Función para preparar datos
# ==========================

def preparar_datos(df_entrada):

    datos = df_entrada.copy()

    # Convertir fecha
    # Transformar la fecha solo si existe
    if "subscription_start_date" in datos.columns:

        datos["subscription_start_date"] = pd.to_datetime(
            datos["subscription_start_date"],
            dayfirst=True,
            errors="coerce"
        )

        datos["subscription_year"] = datos["subscription_start_date"].dt.year
        datos["subscription_month"] = datos["subscription_start_date"].dt.month
        datos["subscription_day"] = datos["subscription_start_date"].dt.day

        datos.drop(columns=["subscription_start_date"], inplace=True)

    # Eliminar columnas igual que en el entrenamiento
    columnas_descartar = [
        "session_id",
        "user_id",
        "movie_id",
        "email",
        "first_name",
        "last_name",
        "title",
        "city",
        "watch_date",
        "created_at",
        "added_to_platform",
        "number_of_seasons",
        "number_of_episodes",
        "production_budget",
        "box_office_revenue",
        "genre_secondary"
    ]

    datos = datos.drop(columns=columnas_descartar)

    # Eliminar la variable objetivo
    if "subscription_plan" in datos.columns:
        datos = datos.drop(columns=["subscription_plan"])

    # One-Hot
    columnas_cat = datos.select_dtypes(include=["object"]).columns
    datos = pd.get_dummies(
        datos,
        columns=columnas_cat,
        drop_first=True,
        dtype=int
    )
    # Agregar columnas faltantes
    for col in columnas_modelo:
        if col not in datos.columns:
            datos[col] = 0

    # Ordenar columnas
    datos = datos[columnas_modelo]
    # Escalar
    datos = scaler.transform(datos)
    return datos



# ==========================
# Menú lateral
# ==========================

pagina=st.sidebar.radio(
"📋 Menú Principal",

[
"🏠 Inicio",
"🎯 Predicción",
"📊 Modelo",
"📘 Acerca"
]
)


# Menú INICIO ->>>>>>

if pagina=="🏠 Inicio":

    st.markdown("""
    # 🎬 Netflix Subscription Predictor

    ### Sistema Inteligente de Predicción del Plan de Suscripción

    ---

    Este sistema utiliza un modelo de **Machine Learning Random Forest**
    para predecir el plan de suscripción de un usuario de Netflix.

    """)

    col1,col2,col3=st.columns(3)

    with col1:

        st.metric(
            "Accuracy",
            "93.25%"
        )

    with col2:

        st.metric(
            "Modelo",
            "Random Forest"
        )

    with col3:

        st.metric(
            "Clases",
            "4"
        )

    st.info("""
    Seleccione una opción desde el menú lateral para comenzar.
    """)
 




# Menú PREDICCION ->>>>>>
elif pagina=="🎯 Predicción":
    st.title("🎯 Predicción del Plan de Suscripción")

    modo = st.radio(
        "Seleccione el modo de predicción:",
        [
            "📂 Predicción desde el Dataset",
            "👤 Predicción manual"
        ]
    )

# SubMenú PREDICCION DATASET ->>>>>>
    if modo == "📂 Predicción desde el Dataset":
        st.subheader("📂 Predicción usando el dataset")
        st.success(f"Dataset cargado correctamente ({len(df)} registros)")
        fila = st.selectbox(
            "Seleccione un registro",
            df.index
        )
        registro = df.iloc[[fila]]
        st.write("### Información del usuario seleccionado")

        col1, col2 = st.columns(2)
        with col1:

            st.metric("Edad", registro["age"].values[0])
            st.metric("País", registro["country"].values[0])
            st.metric("Género", registro["gender"].values[0])
            st.metric("Plan real", registro["subscription_plan"].values[0])

        with col2:
            st.metric("Dispositivo", registro["device_type"].values[0])
            st.metric("Contenido", registro["content_type"].values[0])
            st.metric("Género", registro["genre_primary"].values[0])
            st.metric("Calidad", registro["quality"].values[0])

        if st.button("🔍 Predecir"):
            datos_modelo = preparar_datos(registro)
            pred = modelo.predict(datos_modelo)
            resultado = encoder.inverse_transform(pred)[0]
            probabilidades = modelo.predict_proba(datos_modelo)[0]

            st.divider()
            st.subheader("🎯 Resultado")
            st.markdown(
    f"""
<div style="background:#202020; padding:30px; border-radius:15px; text-align:center; border:2px solid #E50914;">
<h2 style="color:#E50914;">🎯 PLAN PREDICHO</h2>
<h1 style="color:white;">{resultado}</h1>
</div>
""",
    unsafe_allow_html=True
)
            # Si el registro tiene el plan real
            if "subscription_plan" in registro.columns:
                real = registro["subscription_plan"].values[0]
                st.info(f"Plan real: **{real}**")
                if real == resultado:
                    st.success("✅ Predicción correcta")
                else:
                    st.error("❌ Predicción incorrecta")

            st.divider()
            st.subheader("📊 Probabilidad por plan")

            for clase,prob in zip(
                    encoder.classes_,probabilidades):
                st.write(f"**{clase}**")
                st.progress(float(prob))
                st.caption(f"{prob*100:.2f}%")

            
# SubMenú PREDICCION MANUAL ->>>>>>
    elif modo == "👤 Predicción manual":
        st.subheader("👤 Información del Usuario")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Edad", 18, 100, 30)
            gender = st.selectbox("Género", generos)
            country = st.selectbox("País", paises)
            state = st.selectbox("Estado / Provincia", estados)
            monthly_spend = st.number_input(
                "Gasto mensual",
                min_value=0.0,
                value=20.0
            )

            household_size = st.number_input(
                "Personas en el hogar",
                min_value=1,
                max_value=10,
                value=3
            )

        with col2:
            device = st.selectbox("Dispositivo", dispositivos)

            primary_device = st.selectbox(
                "Dispositivo principal",
                dispositivos
            )
            quality = st.selectbox("Calidad", calidades)
            rating = st.selectbox("Clasificación", ratings)
            action = st.selectbox("Acción", acciones)
            location_country = st.selectbox(
                "País de visualización",
                sorted(df["location_country"].dropna().unique())
            )

            st.divider()

        st.subheader("🎬 Información del Contenido")
        col3, col4 = st.columns(2)
        with col3:
            content_type = st.selectbox(
                "Tipo de contenido",
                tipos_contenido
            )
            genre = st.selectbox(
                "Género principal",
                generos_contenido
            )
            language = st.selectbox(
                "Idioma",
                idiomas
            )

        with col4:
            origin = st.selectbox(
                "País de origen",
                paises_origen
            )
            release_year = st.number_input(
                "Año de lanzamiento",
                1950,
                2030,
                2020
            )
            duration = st.number_input(
                "Duración (min)",
                1,
                500,
                120
            )
            st.divider()

        st.subheader("📺 Información de Visualización")
        watch_duration = st.number_input(
            "Tiempo visualizado (min)",
            min_value=0.0,
            value=100.0
        )
        progress = st.slider(
            "Progreso (%)",
            0,
            100,
            80
        )
        download = st.checkbox("¿Descargó el contenido?")
        is_active = st.checkbox(
            "Usuario activo",
            value=True
        )
        netflix_original = st.checkbox(
            "¿Es Netflix Original?"
        )
        warning = st.checkbox(
            "Contenido sensible"
        )

        st.divider()
        st.subheader("📅 Fecha de suscripción")
        fecha = st.date_input("Fecha")

        predecir_manual = st.button(
            "🤖 Predecir Plan",
            use_container_width=True
        )

        if predecir_manual:

            datos = pd.DataFrame({

                "session_id":[0],
                "subscription_start_date":[fecha],
                "user_id":[0],
                "movie_id":[0],
                "watch_date":[fecha],
                "device_type":[device],
                "watch_duration_minutes":[watch_duration],
                "progress_percentage":[progress],
                "action":[action],
                "quality":[quality],
                "location_country":[location_country],
                "is_download":[download],
                "user_rating":[rating],
                "email":[""],
                "first_name":[""],
                "last_name":[""],
                "age":[age],
                "gender":[gender],
                "country":[country],
                "state_province":[state],
                "city":[""],
                "subscription_plan":["Basic"],
                "is_active":[is_active],
                "monthly_spend":[monthly_spend],
                "primary_device":[primary_device],
                "household_size":[household_size],
                "created_at":[fecha],
                "title":[""],
                "content_type":[content_type],
                "genre_primary":[genre],
                "genre_secondary":[""],
                "release_year":[release_year],
                "duration_minutes":[duration],
                "language":[language],
                "country_of_origin":[origin],
                "production_budget":[0],
                "box_office_revenue":[0],
                "number_of_seasons":[0],
                "number_of_episodes":[0],
                "is_netflix_original":[netflix_original],
                "added_to_platform":[fecha],
                "content_warning":[warning]
            })

            datos_modelo = preparar_datos(datos)
            pred = modelo.predict(datos_modelo)
            resultado = encoder.inverse_transform(pred)[0]
            probabilidades = modelo.predict_proba(datos_modelo)[0]
            st.success(f"🎉 Plan predicho: {resultado}")
            st.subheader("📊 Probabilidades")
            for clase, prob in zip(encoder.classes_, probabilidades):
                st.write(clase)
                st.progress(float(prob))
                st.write(f"{prob*100:.2f}%")






# Menú MODELO ->>>>>>
if pagina=="📊 Modelo":

    st.title("📊 Información del Modelo")

    st.metric(
        "Accuracy",
        "93.25 %"
    )

    st.metric(
        "Algoritmo",
        "Random Forest"
    )

    st.write("""
El modelo fue entrenado utilizando Scikit-Learn.

Se aplicó:

- Limpieza de datos
- One-Hot Encoding
- StandardScaler
- Random Forest
    """)



# Menú ACERCA DEL PROYECTO ->>>>>>
if pagina=="📘 Acerca":

    st.title("ℹ️ Acerca del Proyecto")

    st.markdown("""


    ### Este proyecto fue desarrollado para el curso de:

    **Analítica de Datos**

    ### Universidad Señor de Sipán


    ## Integrantes:

    ##### > Linares Barboza Enzo Alonso            
    ##### > Ayasta de la Cruz Giancarlo
    ##### > Perales Cabrera Jhon Carlos


    ## Modelo utilizado:

    ### Random Forest

    ## Objetivo:

    ### Predecir el plan de suscripción de un usuario utilizando técnicas de Machine Learning.

""")