import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

st.title("Prueba de funcionalidades con streamlit")

"""
## Esto es markdown

Prueba ***negrita ***

## Imagenes
"""
st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQIjpCyIss2D64jVPZF6l6gao6SeVMQ25zP3Q&usqp=CAU",width=800)

"""
## Renderizando de codigo de Python 

Primero veamos una lista:
"""
lista= ["Hola",1,2,[-1,1]]
st.write("Lista: ", lista)