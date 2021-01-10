import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
# ------------------------------------------------------- Visualizacion
st.title("¿Cúanto cuesta un _bitcoin_?")

im_bit = "https://s1.eestatic.com/2017/12/07/actualidad/Actualidad_267738694_130078247_1024x576.jpg"

st.image(im_bit, width = 325)


'''
Esta página te brindará información para ayudarte a decidir si invertir o no en un bitcoin, ofreciendote el historico de los precios, una aproximación del precio en el futuro y una medida del riesgo que implica invertir en este tipo de activos. Pero primero veamos algunas definiciones. 
### ¿Qué es el _bitcoin_?
El Bitcoin es la primera divisa digital creada en 2009 por Satoshi Nakamoto (desarrollador anónimo de la criptomoneda bajo este pseudónimo). La idea nació como una solución al problema del doble gasto producido por los intermediarios en una transacción y así, los pagos en línea son enviados directamente entre uno a otro usuario.
Ya que el costo de mediación incrementa los costos de transacción, elimina la posibilidad de pequeñas transacciones, si consideramos un costo más amplio en la pérdida de la habilidad de hacer pagos no-reversibles por servicios no-reversibles, el costo es aún mayor. Esta solución ofrecía una posibilidad de revertir, y con esto mayor confianza, es esta confianza la que hizo que esta moneda ganara popularidad con el tiempo.

Esta moneda ya se puede utilizar de diferentes formas como reservar hoteles en Expedia,
comprar muebles en Overstock, o comprar juegos de Xbox, etc. Pero gran parte de la razón por la 
que se habla tanto del bitcoin es por el intercambio, comprando y vendiendo estos activos 
se pueden generar ganancias. 
'''

# ------------------------------------------------------ Codigo
'''
### ¿Porque invertir en el?
Primero empeza
This is some _markdown_.
'''

df = pd.read_csv('data.csv')
percent = int(df.iloc[-1,-1])/int(df.iloc[0,-1])
percent

price = df.iloc[:,[0,-1]]
price


