# ----------------------------------------------------------------------------- Bibliotecas
import streamlit as st 
import pandas as pd 
import statistics
import plotly.express as px 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from pandas import concat
from bokeh.models.widgets import Div
from tensorflow.keras.models import load_model
# ----------------------------------------------------------------------------- Funciones
model = load_model('modeloFinal.h5', compile = False)
df = pd.read_csv('data_complete.csv')
df.drop(['shangai_stock_exchange(USD)','petroleo(USD)','euro_stoxx50(USD)','dow(USD)'], axis=1, inplace = True)

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Convierte un conjunto de series de tiempo en un conjunto de aprendizaje supervisado

    Params:
        data: secuencia de observaciones
        n_in: tamaño de la ventana, es decir, 2 implica que el siguiente valor
            se calculara con los dos anteriores
        n_out: número de variables a predecir
        dropnan: indica si se eliminarán los valores nulos
    Returns:
    Conjunto de datos de un problema de aprendizaje supervisado
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        # shift
        # se utiliza para desplazar el índice de DataFrame por un número 
        # determinado de períodos con una frecuencia de tiempo opcional
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def get_data(df, ventana):
    '''
    Transforma nuestros datos en un formato aceptable para la red separandolos en train y test

    Params:
      df: conjunto de datos 
      ventana: número de días con el que se predecirá el siguiente
    '''
    n_cols = df.shape[1] - 1 # Numero de variables incluyendo la variable objetivo
    entrenamiento = 800 #Cuantos dias de entrenamiento (restante sera para el conjunto test)
    n_obs = ventana*n_cols
    
    
    values = df.iloc[1082:,1:].values# Desde el primer dia del 2018
    transformer = StandardScaler()
    transformer.fit(values)
    # se escalan los datos para que la red trabaje mejor
    transformer_y = StandardScaler()
    transformer_y.fit(df.iloc[1082:,-1].values.reshape(-1,1))

    values = transformer.transform(values)
    # obtenemos un datos para un problema supervisado
    data = series_to_supervised(values, ventana,1 )
    
    train = data.iloc[:entrenamiento,:]
    test = data.iloc[entrenamiento:,:]

    train_X, train_y = train.iloc[:, :n_obs].values, train.iloc[:, -1].values
    test_X, test_y = test.iloc[:, :n_obs].values, test.iloc[:, -1].values
    
    # modificamos las dimensiones del train y test para que tengan un formato
    # (muestras, ventana, numero de variables)
    train_X = train_X.reshape((train_X.shape[0], ventana, n_cols))
    test_X = test_X.reshape((test_X.shape[0], ventana, n_cols))
    return train_X, train_y, test_X, test_y,transformer,transformer_y

def get_ultimo_dia(df, ventana,transformer,transformer_y):
    '''
    Params:
      df: conjunto de datos 
      ventana: número de días con el que se predecirá el siguiente
    '''
    n_cols = df.shape[1] - 1 # Numero de variables incluyendo la variable objetivo
        
    values = df.iloc[-ventana:,1:].values# Ultima ventana del dataset
    # se escalan los datos para que la red trabaje mejor
    values = transformer.transform(values)
    
    data= values.reshape(-1,n_cols)# se modifica su dimension para que quede coomo una ventana

    data=data.reshape(1,3,n_cols)# se modifica la dimension para que quede como muestra, ventana, num_variables
    return(data)
    
def predecir(x_ultimo,transformer_y,model):
    y_pred= model.predict(x_ultimo)
    y_pred_inv=transformer_y.inverse_transform(y_pred)
    return y_pred_inv[0][0]

def grafica_serie(df):
    df['date'] = df['date'].astype('datetime64')
    fig = px.line(df, x = "date", y = "BCHAIN-MKPRU (USD)") 
    fig = px.area(df, x = "date", y = "BCHAIN-MKPRU (USD)", title = "Histórico del precio del Bitcoin")
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
    st.markdown(f"Incremento: {round(p,2)} %")
    
def get_riesgo(df):
    dates = list(df['date'])
    dates = [x.strftime('%Y-%m-%d') for x in dates]
    st.markdown("Rango de fecha:")
    col1 = st.selectbox('Inicio:', dates)
    idx1 = dates.index(col1)
    col2 = st.selectbox('Final:', dates[idx1 + 4:])
    idx2 = dates.index(col2)
    riesgo = round(statistics.stdev(df.iloc[idx1:idx2,-1]),3)
    st.markdown(f"Riesgo: {riesgo} USD")
    
def get_media(df):
    dates = list(df['date'])
    dates = [x.strftime('%Y-%m-%d') for x in dates]
    st.markdown("Rango de fecha:")
    col1 = st.selectbox('Inicio:  ', dates)
    idx1 = dates.index(col1)
    col2 = st.selectbox('Final:  ', dates[idx1 + 4:])
    idx2 = dates.index(col2)
    riesgo = round(df.iloc[idx1:idx2,-1].mean(),3)
    st.markdown(f"Precio medio: {riesgo} USD")
    
def plot_xy(df, test_X, y_pred_inv):
    fig = plt.figure(figsize=(6,5))
    plt.plot(range(0,len(test_X)),df.iloc[-len(test_X):,-1],color="turquoise",label="Real")
    plt.plot(range(0,len(test_X)),y_pred_inv, color="lightcoral",label="Predicha")
    plt.xlabel("días")
    plt.ylabel("USD")
    plt.title("Predicción del último periodo")    
    return fig

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
st.markdown("<h1 style='color: white;'>Pricing de Bitcoin</h1>",unsafe_allow_html=True)

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
    if st.button('¿Que es un Block chain?'):
        js = "window.open('https://economiatic.com/blockchain/')"  # New tab or window
        js = "window.location.href = 'https://economiatic.com/blockchain/'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        bloc = Div(text=html)
        st.bokeh_chart(bloc) 

#st.header("¿Cúanto cuesta un _bitcoin_?")
st.markdown("<h2 style='text-align: center; color: yellow;'>¿Cúanto cuesta un bitcoin?</h2>",unsafe_allow_html=True)
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

st.markdown("<h2 style='text-align: center; color: yellow;'>El riesgo de la criptomoneda</h2>",unsafe_allow_html=True)

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

Por ejemplo en la semana de del 25 al 31 de Diciembre de 2020, el precio del bitcoin aumento en un 
34.34 %, y el riesgo durante esa semana fue de 2471.651, es decir los precios del bitcoin en esa
semana estan alejados 2471.651 USD de la media aritmética de los mismos. Esto nos da una idea de que 
tanto varían los precios durante un periodo de tiempo y con ayuda de la predicción al precio del bitcoin 
podemos tener un panorama más claro de la situación.
'''
st.markdown("<h2 style='text-align: center; color: orange;'>Porcentaje en el aumento de precios</h2>",unsafe_allow_html=True)
incremento(df)

st.markdown("<h2 style='text-align: center; color: orange;'>Media aritmética del precio</h2>",unsafe_allow_html=True)
get_media(df)

st.markdown("<h2 style='text-align: center; color: orange;'>Riesgo en los precios</h2>",unsafe_allow_html=True)
get_riesgo(df)

st.markdown("<h2 style='text-align: center; color: yellow;'>Predicciones</h2>",unsafe_allow_html=True)
train_X, train_y, test_X, test_y,transformer ,transformer_y = get_data(df, 3)    
x_ultimo = get_ultimo_dia(df, 3, transformer, transformer_y)
prediction = round(float(predecir(x_ultimo,transformer_y,model)),3)

y_pred = model.predict(test_X)
y_pred_inv = transformer_y.inverse_transform(y_pred)
st.plotly_chart(plot_xy(df, test_X, y_pred_inv))

'''
Ya que la volatilidad en los precios del Bitcoin es muy alta, y debido a aquellas circunstancias no medibles
antes mensionadas se predice el precio que tendrá el bitcoin el día siguiente, pues las predicciones en días posteriores
podrían estar muy alejadas del valor real.
'''
if (st.button("Predecir")):
    st.markdown(f"<h2 style='text-align: center; color: white;'>{prediction} USD</h2>",unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: yellow;'>¿Como obtener bitcoins?</h2>",unsafe_allow_html=True)
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
    
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------- Barra lateral
k = "Kenia Cisneros"
r = "Rodrigo Diaz"
#st.sidebar.markdown(r)
#st.sidebar.markdown(k)

st.sidebar.title ("Toma decisiónes de inversión inteligentes")

barra="Esta página te brindará información para ayudarte a decidir si invertir o no en un bitcoin, ofreciendote el historico de los precios, una aproximación del precio en el futuro y una medida del riesgo que implica invertir en este tipo de activos."

st.sidebar.markdown(f"<h4 style= color: balck;'>{barra}</h4>",unsafe_allow_html=True)

esp = "La estimación del precio no es ''exacto'' debido a que existen circunstancias que afectan a esta divisa y no son medibles. Sin embargo, es útil saber una estimación aproximada del precio."
st.sidebar.markdown(f"<h4 style= color: balck;'>{esp}</h4>",unsafe_allow_html=True)

st.sidebar.title("¿Qué es el _bitcoin_?")
bitcoin = "El Bitcoin es la primera divisa digital creada en 2009 por Satoshi Nakamoto. Como una solución al problema del doble gasto producido por los intermediarios en una transacción."
st.sidebar.markdown(f"<h4 style= color: balck;'>{bitcoin}</h4>",unsafe_allow_html=True)

st.sidebar.title("¿Qué es el _riesgo_?")
riesgo = "Posibilidad de que el rendimiento real de una inversión difiera de lo esperado."
st.sidebar.markdown(f"<h4 style= color: balck;'>{riesgo}</h4>",unsafe_allow_html=True)
