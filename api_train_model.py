import pandas as pd
import numpy as np
import matplotlib.pylab as plt

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('fast')

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data.csv', parse_dates=[0],index_col=0, squeeze=True)
print(df.head())
df['weekday']=[x.weekday() for x in df.index]
df['month']=[x.month for x in df.index]
print("Dataframe")
print(df.head())

print("Primer dato y Utimo dato por fecha")
print(df.index.min())
print(df.index.max())
#Y ahora veamos cuantas muestras tenemos de cada año
print("Muestras por año (2019-2020)")
print(len(df['2019']))
print(len(df['2020']))
#datos estadísticos que nos brinda pandas
print("Datos Estadisticos")
print(df.describe())
#Promedios Mensuales
print("Promedios Mensuales")
meses =df.resample('M').mean()
print(meses)
#Visualizamos las medias Mensuales
print("Grafica Medias Mensuales")
plt.plot(meses['2019'].values)
plt.plot(meses['2020'].values)

#Ventas Diarias Junio y julio
print("Grafica de ventas diarias (en unidades) en junio y julio")
verano2019 = df['2019-06-01':'2019-09-01']
plt.plot(verano2019.values)
verano2020 = df['2020-06-01':'2020-09-01']
plt.plot(verano2020.values)

###################REDES NEURONALES############################
print("###################REDES NEURONALES############################")
print("Convertimos en un “problema de tipo supervisado")
PASOS=7

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
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
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
# load dataset
values = df.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, PASOS, 1)
print(reframed.head())
print("###################CREACION DE LA RNA############################")
#####CREACION#########
# split into train and test sets
values = reframed.values
n_train_days = 315+180 - (30+PASOS)
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
x_train, y_train = train[:, :-1], train[:, -1]
x_val, y_val = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

#tangente hiperbolica
def crear_modeloFF():
    model = Sequential() 
    model.add(Dense(PASOS, input_shape=(1,PASOS),activation='tanh'))
    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_absolute_error',optimizer='Adam',metrics=["mse"])
    model.summary()
    return model
print("################### ENTRENAMIENTO Y RESULTADOS ############################")

EPOCHS=40

model = crear_modeloFF()

history=model.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_val,y_val),batch_size=PASOS)
print("Grafica Conjunto de validación (30 días)")
results=model.predict(x_val)
plt.scatter(range(len(y_val)),y_val,c='g')
plt.scatter(range(len(results)),results,c='r')
plt.title('validate')
plt.show()
print("Pronóstico de ventas futuras (Ultimas semanas de Julio)")
ultimosDias = df['2020-7-12':'2020-7-28']
ultimosDias

values = ultimosDias.values
values = values.astype('float32')
# normalize features
values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, PASOS, 1)
reframed.drop(reframed.columns[[7]], axis=1, inplace=True)
reframed.head(7)

values = reframed.values
x_test = values[5:, :]
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
x_test

def agregarNuevoValor(x_test,nuevoValor):
    for i in range(x_test.shape[2]-1):
        x_test[0][0][i] = x_test[0][0][i+1]
    x_test[0][0][x_test.shape[2]-1]=nuevoValor
    return x_test

results=[]
for i in range(7):
    parcial=model.predict(x_test)
    results.append(parcial[0])
    print(x_test)
    x_test=agregarNuevoValor(x_test,parcial[0])

adimen = [x for x in results]    
inverted = scaler.inverse_transform(adimen)
print(inverted)

prediccion1SemanaAgosto = pd.DataFrame(inverted)
prediccion1SemanaAgosto.columns = ['pronostico']
print("Pronostico para la primera semana de Agosto")
print(prediccion1SemanaAgosto)
prediccion1SemanaAgosto.plot()
prediccion1SemanaAgosto.to_csv('pronostico.csv')