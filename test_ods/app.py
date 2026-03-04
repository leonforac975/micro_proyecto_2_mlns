# Librerias
import os
import streamlit as st
import joblib
import numpy as np
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# Nombres de los ODS
ODS_nombres = {
    1:  "Fin de la pobreza",
    2:  "Hambre cero",
    3:  "Salud y bienestar",
    4:  "Educación de calidad",
    5:  "Igualdad de género",
    6:  "Agua limpia y saneamiento",
    7:  "Energía asequible y no contaminante",
    8:  "Trabajo decente y crecimiento económico",
    9:  "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
    17: "Alianzas para lograr los objetivos",
}

# Preprocesamiento
tokenizer = RegexpTokenizer(r'\w+')
stemmer   = SnowballStemmer(language='spanish')
stop      = set(stopwords.words('spanish'))

def text_preprocess(text):
    text   = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [w for w in tokens if w not in stop]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

# Cargar modelo
@st.cache_resource # Se hace para no cargar el modelo en cada interacción
def cargar_modelo():
    ruta = os.path.join(os.path.dirname(__file__), 'pipeline_svm_ods.pkl')
    search = joblib.load(ruta)
    return search.best_estimator_

# Interfaz
col1, col2 = st.columns([1, 4])

with col1:
    logo_path = os.path.join(os.path.dirname(__file__), 'src', 'ODS_LOGO.jpg')
    if os.path.exists(logo_path):
        st.image(logo_path)

with col2:
    st.title("Clasificador de Objetivos de Desarrollo Sostenible")
    st.write("Desarrollado por Ricardo Gutierrez y Leonardo Forero")
    st.write("Microproyecto 2 - Machine Learning No Supervisado")
    st.write("Maestría en Inteligencia Artificial - Universidad de los Andes")
    st.write(" ")

texto = st.text_area("Ingresa un texto en español y el modelo identificará a qué ODS pertenece.", height=150, placeholder="Escribe o pega aquí el texto...")

if st.button("Clasificar"):
    if not texto.strip():
        st.warning("Por favor ingresa un texto antes de clasificar.")
    else:
        pipeline = cargar_modelo()
        texto_proc = text_preprocess(texto)
        pred       = pipeline.predict([texto_proc])[0]
        probs      = pipeline.predict_proba([texto_proc])[0]
        clases     = pipeline.classes_
        top3_idx   = np.argsort(probs)[::-1][:3]

        st.success(f"ODS {pred} — {ODS_nombres.get(pred, 'Desconocido')}")

        imagen_path = os.path.join(os.path.dirname(__file__), 'src', f'ODS_{pred:02d}.jpg')
        if os.path.exists(imagen_path):
            st.image(imagen_path)
        else:
            st.warning(f"No se encontró la imagen para ODS {pred}")
