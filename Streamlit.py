import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
# ------------------------------------------------------- Visualizacion
st.title("¿Cúanto cuesta un _bitcoin_?")

im_bit = "https://s1.eestatic.com/2017/12/07/actualidad/Actualidad_267738694_130078247_1024x576.jpg"

st.image(im_bit, width = 325)


'''
Esta página te ayuda a predecir el precio del _bitcoin_, no ...
### ¿Qué es el _bitcoin_?
Bitcoin es una nueva moneda que fue creada en 2009 por una persona desconocida 
usando el alias Satoshi Nakamoto. Las transacciones se realizan sin intermediarios,
 es decir, ¡sin bancos! Bitcoin se puede utilizar para reservar hoteles en Expedia,
 comprar muebles en Overstock y comprar juegos de Xbox. Pero gran parte de la razón
 por la que se habla tanto del bitcoin se trata de hacerse rico mediante el 
 intercambio. La criptomoneda aumentó un 9% a un nuevo máximo histórico de 
 aproximadamente 19.860 dólares este lunes. Lo que superó el récord anterior 
 de 19.783 dólares en diciembre de 2017.

'''

# ------------------------------------------------------ Codigo
'''
### ¿Porque invertir en el?
Primero empeza
This is some _markdown_.
'''

df = pd.read_csv('data.csv')
price = df.iloc[:,-1]
plt.fill_between( x, y, color="skyblue", alpha=0.2)
plt.plot(x, y, color="Slateblue", alpha=0.6)

fig = sns.relplot(data = tips, x = "total_bill", y = "tip", col = "time", hue = "smoker", size = "size")
st.pyplot(fig)

"""

## Renderizando de codigo de Python 

Primero veamos una lista:
"""
lista= ["Hola",1,2,[-1,1]]






















st.write("Lista: ", lista)