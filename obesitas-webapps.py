import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.naive_bayes import GaussianNB

st.write("""
# Aplikasi deteksi Obesitas
Aplikasi berbasis web ini memprediksi(mengklsidikasi) penyakit obesitas .
Data diperoleh dari [kaggle](https://www.kaggle.com/datasets/sujithmandala/obesity-classification-dataset).
""")

img = Image.open('obes.jpg')
img = img.resize((610, 418))
st.image(img,use_column_width=False)

st.sidebar.header('Parameter Inputan')

# menu upload csv inputan
upload_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])
if upload_file is not None:
    inputan = pd.read_csv(upload_file)
else:
    def input_user():
        Age = st.sidebar.slider('Age (Year)', 10,120,40)
        Gender = st.sidebar.selectbox('Gender',('Male','Female'))
        Height = st.sidebar.slider('Height (cm)', 100,230,170)
        Weight = st.sidebar.slider('Weight (Kg)', 20,200,50)
        BMI = st.sidebar.slider('BMI ', 3.0,40.0,21.0)
        data = {'Age' : Age,
                'Gender' : Gender,
                'Height' : Height,
                'Weight' : Weight,
                'BMI' : BMI
                }
        fitur = pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()

# menggabungkan inputan dan dataset obesitas
obesitas_raw = pd.read_csv('obesitasclean.csv')
obesitas = obesitas_raw.drop(columns=['Label'])
df = pd.concat([inputan, obesitas], axis=0)

# encode untuk fitur ordinal 
encode = ['Gender']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]  #ambil baris pertaman (input data user)

# menampilkan parameter hasil inputan
st.subheader('Parameter inputan')

if upload_file is not None:
    st.write(df)
else:
    st.write('Menunggu upload file CSV. Saat ini memakai sample inputan (Seperti tampilan dibawah)')
    st.write(df)

# memuat model NBC
load_model = pickle.load(open('modelNBC_obesitas.pkl', 'rb'))

# Prediksi
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

st.subheader('Keterangan Label Kelas')
obe_jenis = np.array(['Underweight', 'Normal Weight', 'Overweight', 'Obese'] )
st.write(obe_jenis)

st.subheader('Hasil prediksi (klasifikasi obesitas)')
st.write(obe_jenis[prediksi])

st.subheader('Probabilitas Hasil Prediksi (Klasifikasi Obesitas)')
st.write(prediksi_proba)