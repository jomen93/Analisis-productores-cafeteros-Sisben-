# ============================================================================
# Archivo cnn_sisben.py
# autor Johan S. Mendez, Jose D. Mendez
# fecha 27/Agos/2020

# Clasificacion de beneficiarios del nuevo sistema de clasificacion del sisben
# agrupado en 4 grandes grupos de beneficiarios, se utiliza un red neuronal
# simple para el primero modelo de clasificación de salida multiple


# ============================================================================


import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
# from keras.callbacks import ModelCheckpoint
# from keras.models import model_from_json
# from keras import backend as K
# from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


print("\033[91m Lectura de datos \033[0m")
#************ Preparing the data ************#

# Read the data
path = "/Users/johan/Documents/GitHub/Cafe/bases_datos/base_maestra_sisben_iv.csv"
df = pd.read_csv(path)
# Eliminate the missing values for dataset
df = df.dropna()
# Choose a subset of the complete data
df_sample = df.sample(n=150000, random_state=123)

# Preparing the data to the model
X = df_sample.drop("sisben_iv", axis=1)
y = df_sample["sisben_iv"]
y = pd.get_dummies(y, dtype=float)

# We separate the categorical and numerical data in diferente variables
X_categorical = X.select_dtypes(include=["int", "object"])
X_categorical = pd.get_dummies(X,
                               columns=X_categorical.columns,
                               dtype=float)

x_train, x_test, y_train, y_test = train_test_split(np.asarray(X_categorical),
                                                    np.asarray(y),
                                                    test_size=0.33,
                                                    shuffle=True)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model = Sequential()

model.add(Conv1D(64, (32),
          input_shape=(x_train.shape[1], 1),
          activation='elu'))
model.add(Flatten())
model.add(Dense(32, activation='elu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer="Adam",
              metrics=[tf.keras.metrics.CategoricalAccuracy()],
              )

print(model.summary())

batch_size = 128
epochs = 100
model = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))


plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.plot(model.history['categorical_accuracy'])
plt.plot(model.history['val_categorical_accuracy'])
plt.title('model train vs validation categorical_accuracy')
plt.ylabel('categorical_accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
