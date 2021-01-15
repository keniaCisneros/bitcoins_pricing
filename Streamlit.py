# ----------------------------------------------------------------------------- Bibliotecas
import streamlit as st 
import pandas as pd 
import numpy as np 
import statistics
import plotly.express as px 
from plotly.subplots import make_subplots 
import plotly.graph_objects as go 
import matplotlib.pyplot as plt
from bokeh.models.widgets import Div
# ----------------------------------------------------------------------------- Funciones

df = pd.read_csv('data_complete.csv')

def grafica_serie(df):
    df['date'] = df['date'].astype('datetime64')
    fig = px.line(df, x = "date", y = "BCHAIN-MKPRU (USD)") 
    fig = px.area(df, x = "date", y = "BCHAIN-MKPRU (USD)")
    plt.title("Title")
    st.plotly_chart(fig)
    
def incremento_percent(v1, v2):
    '''
    Esta función calcula el incremento en porcentaje de un precio
    '''
    p = ((v2 - v1)/v1)*100
    return p

def incremento(df):
    dates = df['date']
    dates = list(dates.apply(lambda x: x.strftime('%Y-%m-%d')))
    st.markdown("Rango de fecha:")
    col1 = st.selectbox('Inicio: ', dates)
    idx1 = dates.index(col1)
    col2 = st.selectbox('Final: ', dates[idx1 + 1:])
    idx2 = dates.index(col2)
    v1 = int(df.iloc[idx1,-1])
    v2 = int(df.iloc[idx2,-1]) 
    p = incremento_percent(v1, v2)
    st.markdown(f"El incremento en el precio del Bitcoin de {col1} a {col2}, es {round(p,2)} %")
    
def get_riesgo(df):
    dates = list(df['date'])
    dates = [x.strftime('%Y-%m-%d') for x in dates]
    st.markdown("Rango de fecha:")
    col1 = st.selectbox('Inicio:', dates)
    idx1 = dates.index(col1)
    col2 = st.selectbox('Final:', dates[idx1 + 4:])
    idx2 = dates.index(col2)
    riesgo = round(statistics.stdev(df.iloc[idx1:idx2,-1]),3)
    st.markdown(f"Riesgo de {col1} a {col2}, es {riesgo}")

# ----------------------------------------------------------------------------- Estructura de la página
page_bg_img = '''
<style>
body {
background-image: url("https://w.wallhaven.cc/full/47/wallhaven-477m53.png");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
#st.title("_Bitcoins_")
st.markdown("<h1 style='color: black;'>Pricing de Bitcoin</h1>",unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: red;'>Advertencia</h2>",unsafe_allow_html=True)
st.markdown(
'''
<style>p{color:white;}</style>
<p>
El Bitcoin es una nueva moneda experimental que está en desarrollo activo, que  con el tiempo ha estado creciendo con su uso constante. 
Como tal,su valor en el futuro no se puede predecir con exactitud  debido a que existen 
circunstancias externas que no son medibles, como son las sociales, politicas, globales o eventos
inesperados como la presente pandemia. 

Sin embargo, es posible tener un precio aproximado al precio que tendrá el bitcoin, utilizando variables
que afectan la volsa de valores como el precio del petroleo, oro, etc. y variables relacionadas con 
los Block chains de las criptomonedas, pues la cantidad de bitcoins disponibles depende de ello. 
Es por eso que el modelo solo dara una buena estimacion bajo circunstancias normales y no contemplará estos cambios externos antes mencionados



Si aún no estas tan familiarizado con lo que es un Bitcoin o un Block chain puedes pulzar los botones 
y te redirigiremos a una página con estos detalles.
</p>
''',unsafe_allow_html=True)
cb1, mid1, cb2 = st.beta_columns([5,5,5])
with cb1:
    if st.button('¿Que es un bitcoin?'):
        js = "https://www.eleconomista.com.mx/mercados/Que-son-los-Bitcoins-20170524-0105.html')"  # New tab or window
        js = "https://www.eleconomista.com.mx/mercados/Que-son-los-Bitcoins-20170524-0105.html'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        bit = Div(text=html)
        st.bokeh_chart(bit)
with cb2:
    if st.button('¿Que es un Bloch chain?'):
        js = "window.open('https://economiatic.com/blockchain/')"  # New tab or window
        js = "window.location.href = 'https://economiatic.com/blockchain/'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        bloc = Div(text=html)
        st.bokeh_chart(bloc) 

#st.header("¿Cúanto cuesta un _bitcoin_?")
st.markdown("<h2 style='text-align: center; color: green;'>¿Cúanto cuesta un bitcoin?</h2>",unsafe_allow_html=True)
text = "Gracias a que la popularidad de esta divisa ha ido en aumento, son cada vez más personas las que invierten en ella, y esto aunado a los eventos sociales tiene como consecuencia la volatilidad de los precios, ya que sí en algún momento los precios bajan entonces hay una venta masiva."
im_bit = "https://s1.eestatic.com/2017/12/07/actualidad/Actualidad_267738694_130078247_1024x576.jpg"

#------------------------------------------------------------------------------ Imagen con texto a un lado
col1, mid, col2 = st.beta_columns([20,10,20])
with col1:
    st.image(im_bit, width=400)
with col2:
    st.write(text)
#---------------------------------------------------------------------------------------------------------

'''
Con ayuda de la siguente gráfica puedes explorar los datos históricos que ha tenido el bitcoin, 
examinando solamente un periodo o el histórico completo para que te des una idea de la volatilidad
de esta moneda y la tomes en cuenta al momento de tomar una decisión.
'''
grafica_serie(df)

st.markdown("<h2 style='text-align: center; color: green;'>El riesgo de la criptomoneda</h2>",unsafe_allow_html=True)
#st.header("El riesgo de la criptomoneda")
link = '[GitHub](http://github.com)'

'''
Un factor importante a considerar al momento de tomar una decisión es el riesgo en el tiempo sobre
el precio del bitcoin. Si no estas familiarizado con este término o no sabes por que es importante 
considerarlo puedes presionar el siguiente botón.
'''
if st.button('¿Por que considerar el Riesgo?'):
    js = "window.open('https://www.finanzaspracticas.com.mx/finanzas-personales/scervicios-bancarios/ahorro-e-inversiones/el-riesgo-financiero-en-el-tiempo?print=y')"  # New tab or window
    js = "window.location.href = 'https://www.finanzaspracticas.com.mx/finanzas-personales/scervicios-bancarios/ahorro-e-inversiones/el-riesgo-financiero-en-el-tiempo?print=y'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    risk = Div(text=html)
    st.bokeh_chart(risk)

'''
Te presentamos una medida de riesgo que relaciona los precios obtenidos 
con los esperados mostrando que tan alejados estan unos de otros. Puedes elegir un rango de tiempo 
del que deseas saber el riesgo, igualmente podrías preguntarte en que porcentaje aúmento o disminuyó
el precio.

Por ejemplo en la semana de del 26 al 30 de Diciembre de 2020, el precio del bitcoin aumento en un 
14%, y el riesgo durante esa semana fue de 1301.112, es decir ....
'''
st.markdown("<h2 style='text-align: center; color: green;'>Porcentaje en el aumento de precios</h2>",unsafe_allow_html=True)
#st.subheader('Porcentaje en el aumento de precios')
incremento(df)
st.markdown("<h2 style='text-align: center; color: green;'>Riesgo en los precios</h2>",unsafe_allow_html=True)
#st.subheader('Riesgo en los precios')
get_riesgo(df)
st.markdown("<h2 style='text-align: center; color: green;'>¿Como obtener bitcoins?</h2>",unsafe_allow_html=True)
#st.header("¿Como obtener bitcoins?")

'''
Si es que te has decidido a adquirir una de estas monedas, podrías preguntarte donde, 
aquí te mostramos algunas páginas donde puedes adquirirlos. Son plataformas que cuentan
con una gran base de usuarios y en las que puedes pagar con tu tarjeta de crédito o a
través de transferencia bancaria:
    
Coinbase: ofrece una comisión del 3.5%, pero es uno de los sitios más populares.

Kraken: la comisión es mucho menor que con Coinbase, entre el 0.16 y 0.26%.

LocalBitcoins: para la compra de persona a persona (P2P)
'''

# ----------------------------------------------------------------------------- botones 
    
c1, mid1, c2, mid2, c3 = st.beta_columns([5,5,5,5,5])
with c1:
    if st.button('Coinbase'):
        js = "window.open('https://www.coinbase.com/es-LA/')"  # New tab or window
        js = "window.location.href = 'https://www.coinbase.com/es-LA/'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        coin = Div(text=html)
        st.bokeh_chart(coin)
with c2:
    if st.button('Kraken'):
        js = "window.open('https://www.kraken.com/')"  # New tab or window
        js = "window.location.href = 'https://www.kraken.com/'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        kra = Div(text=html)
        st.bokeh_chart(kra) 
with c3:
    if st.button('LocalBitcoins'):
        js = "window.open('https://localbitcoins.com/es/')"  # New tab or window
        js = "window.location.href = 'https://localbitcoins.com/es/'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        local = Div(text=html)
        st.bokeh_chart(local)
    
# ----------------------------------------------------------------------------- Barra lateral
k = "Kenia Cisneros"
r = "Rodrigo Diaz"
#st.sidebar.markdown(r)
#st.sidebar.markdown(k)
barra = "Esta página te brindará información para ayudarte a decidir si invertir o no en un bitcoin, ofreciendote el historico de los precios, una aproximación del precio en el futuro y una medida del riesgo que implica invertir en este tipo de activos."
st.sidebar.title ("Toma decisiónes de inversión inteligentes")
st.sidebar.markdown(barra)

esp = "La estimación del precio no es ''exacto'' debido a que existen circunstancias que afectan a esta divisa y no son medibles. Sin embargo, es útil saber una estimación aproximada del precio."
st.sidebar.markdown(esp)

st.sidebar.title("¿Qué es el _bitcoin_?")
bitcoin = "El Bitcoin es la primera divisa digital creada en 2009 por Satoshi Nakamoto. Como una solución al problema del doble gasto producido por los intermediarios en una transacción."
st.sidebar.markdown(bitcoin)

st.sidebar.title("¿Qué es el _riesgo_?")
bitcoin = "Posibilidad de que el rendimiento real de una inversión difiera de lo esperado."
st.sidebar.markdown(bitcoin)

#link = '[GitHub](http://github.com)'
#st.sidebar.markdown(link, unsafe_allow_html=True)