# ============================================================================
# Archivo cnn_sisben_nibario.py
# autor Johan S. Mendez, Jose D. Mendez
# fecha 27/Agos/2020

# Clasificacion de beneficiarios del sisben a partir de una sencilla red
# neuronal.
# ============================================================================


import tensorflow as tf
from tensorflow import set_random_seed
set_random_seed(2)
import numpy as np
from numpy.random import seed
seed(1)
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
path = "/Users/johan/Documents/GitHub/Cafe/bases_datos/base_calculos.csv"
df = pd.read_csv(path)
df.head(5)
df = pd.DataFrame(data=df)
df_sample = df.sample(frac=0.1, random_state=0)

independientes = df_sample[["zona", "discapa", "nivel", "edad", "teneviv",
                            "pared", "piso", "energia", "energia", "alcanta",
                            "gas", "telefono", "basura", "acueduc",
                            "elimbasura", "sanitar", "ducha", "llega",
                            "cocinan", "preparan", "alumbra", "usotele",
                            "nevera", "lavadora", "tvcolor", "tvcable",
                            "calenta", "horno", "aire", "computador",
                            "equipo", "moto", "tractor", "auto1", "bieraices",
                            "area_total"]]

independientes = pd.get_dummies(independientes,
                                columns=independientes.columns,
                                dtype=float)

x = independientes.to_numpy()
y = df_sample["subsidiado"].to_numpy()

print("\033[91m size of  data ------->{} \033[91m".format(np.shape(x)))


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=123)


print("\033[91m size train data ----->{} \033[91m".format(np.shape(x_train)))
print("\033[91m size validation data ->{} \033[91m".format(np.shape(x_test)))

#layer_input = keras.Input(shape = (Xx_train.shape[1],))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model = Sequential()

model.add(Conv1D(int(x_train.shape[1]/128),
          (8),
          input_shape=(x_train.shape[1], 1),
          activation='relu'))

model.add(Flatten())
model.add(Dense(16, activation='tanh', kernel_regularizer='l1'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer="Adam",
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])

print(model.summary())

stop = EarlyStopping(patience=10, monitor='val_loss')
history = model.fit(x_train,
                    y_train,
                    batch_size=2048,
                    epochs=200,
                    callbacks=[stop],
                    validation_data=(x_test, y_test),
                    verbose = 1,
                    )

print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=256, verbose=0)
print("test loss: ", results[0])
print("test acc: ", results[1])

y_pred = model.predict(x_test)

for i in range(20):
    print("predicho =", y_pred[i], "   ", "real = ", y_test[i])

print(history.history["loss"])

np.savetxt('loss_cnn.txt', history.history["loss"], delimiter=',')
np.savetxt('val_loss_cnn.txt', history.history["val_loss"], delimiter=',')
np.savetxt('ba_cnn.txt', history.history["binary_accuracy"], delimiter=',')
np.savetxt('val_ba_cnn.txt',
           history.history["val_binary_accuracy"], delimiter=',')

np.savetxt('y_test_cnn.txt', y_test, delimiter=',')
np.savetxt('y_pred_cnn.txt', y_pred, delimiter=',')
