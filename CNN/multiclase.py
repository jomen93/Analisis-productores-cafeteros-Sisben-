# ============================================================================
# Archivo multiclase.py
# autor Johan S. Mendez, Jose D. Mendez
# fecha 27/Agos/2020

# Modelo de redes neuronales de ramas paralelas para predecir la clasificacion
# de los cuatro grupos del sisben


# AUN EN DESARROLLO

# ============================================================================
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
#from keras.utils.vis_utils import plot_model
# from keras.callbacks import EarlyStopping
# from keras.models import model_from_json

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


print("\033[91m Lectura de datos \033[0m")
#************ Preparing the data ************#

# Lectura de datos
path = "/Users/johan/Documents/GitHub/Cafe/bases_datos/base_maestra_sisben_iv.csv"
df = pd.read_csv(path)
# Eliminacion de los valores vacios de la base de datos
df = df.dropna()
# Choose a subset of the complete data
df_sample = df.sample(frac=1, random_state=123)

# Preparacion de los datos para llevarlos al modelo
X = df_sample.drop("sisben_iv", axis=1)
y = df_sample["sisben_iv"]
y = pd.get_dummies(y, dtype=float)

# Normalize numerical data
X_numerical = X.select_dtypes(include=["float"])
X_numerical = X_numerical.drop("bieraices_SISBEN", axis=1)
norm = X_numerical.max()
X_numerical = X_numerical/norm

# We separate the categorical and numerical data in diferente variables
X_categorical = X.select_dtypes(include=["int", "object"])
X_categorical = pd.get_dummies(X, columns=X_categorical.columns, dtype=float)

X_c = X_categorical.shape
X_n = X_numerical.shape

print("\033[91m size of categorical data --->{} \033[91m".format(X_c))
print("\033[91m size of numerical data   --->{} \033[91m".format(X_n), end="")

# La columna "target". Indica el grupo de la persona con cuatro clases
# vamos a separar los daros en cojuntos de validacion y entrenamiento

X_train_n, X_test_n, y_train, y_test = train_test_split(X_numerical,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=123)
X_train_c, X_test_c, y_train, y_test = train_test_split(X_categorical,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=123)
print("**%d entrenamiento %d validacion **" % (len(X_train_c), len(X_test_c)))


# Definiendo los dos conjunto de entrada
categorical_input = keras.Input(shape=(X_c[1],))
numerical_input = keras.Input(shape=(X_n[1],))

# Primera rama, opera la primera entrada
keras.layers.Dropout(0.2, noise_shape=None, seed=123)
x = layers.Dense(32, activation="elu", name="layer1")(categorical_input)
# x = layers.Dense(16, activation="exponential", name = "layer2")(x)
#x = layers.Dense(392*2, activation="sigmoid", name = "layer2")(x)
#x = layers.Dense(392, activation="sigmoid", name = "layer3")(x)
# x = layers.Dense(198, activation="sigmoid", name = "layer4")(x)
# x = layers.Dense(99, activation="sigmoid", name = "layer5")(x)
# x = layers.Dense(33, activation="sigmoid", name = "layer6")(x)
# x = layers.Dense(11, activation="sigmoid", name = "layer7")(x)
outputs = layers.Dense(4, activation="softmax", name="predictions")(x)
#x = keras.Model(inputs=categorical_input, outputs=x)

# Segunda rama, opera sobre el segundo input
y = layers.Dense(16, activation="relu")(numerical_input)
y = layers.Dense(8, activation="relu")(y)
y = layers.Dense(4, activation="softmax")(y)
y = keras.Model(inputs=numerical_input, outputs=y)

# Combiancion de las salidas de las dos ramas
# combined = tf.keras.layers.concatenate([x.output, y.output])


#z = layers.Dense(4, activation = "sigmoid")(combined)
#z = layers.Dense(4, activation = "softmax")(z)

# El modelo acepta las entradas de dos ramas y la salida es un valor unico
model = keras.Model(inputs=categorical_input,
                    outputs=outputs,
                    name="sisben_network_2")

# Falta instalar las librerias correspondientes para poder imprimir la red
# neuronal pero para resolver el problema se puede correr el modelo en colab
# y en este lugar si se puede generar la grafica de la red neuronal
# plot_model(model, to_file="model.png",show_shapes=True,show_layer_names=True)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01,
                                                 beta_1=0.9,
                                                 beta_2=0.999,
                                                 epsilon=1e-07,
                                                 amsgrad=False,
                                                 name="Adam"),
              metrics=[tf.keras.metrics.CategoricalAccuracy()],
              )

#print(model.summary())
print('# Fit model on training data')
print('')
#stop=EarlyStopping(patience=10,monitor='val_accuracy')
#                   callbacks=[stop],

history = model.fit([X_train_c, X_train_n],
                    y_train,
                    batch_size=256,
                    epochs=20,
                    validation_data=([X_test_c, X_test_n], y_test),
                    verbose = 0,
                    )

# serialize model to JSON
model_json = model.to_json()
with open("sisben_network_2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("sisben_network_2_weights.h5")
print("Saved model to disk")

# Evaluate the model on the test data using 'evaluate'
print("Evaluate on test data")
results = model.evaluate([X_test_c, X_test_n],
                         y_test,
                         batch_size=256,
                         verbose=0)
print("test loss: ", results[0])
print("test acc: ", results[1])
    
train = history.history["categorical_accuracy"]
val = history.history["val_categorical_accuracy"]

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Error plots
plt.plot(train, "r", label="entrenamiento")
plt.plot(val, "b", label="validacion")
plt.grid(True)
plt.legend()
plt.savefig("accuracy")

#plt.plot(train_loss,"r",label = "entrenamiento")
plt.plot(val_loss, "b", label="validacion")
plt.grid(True)
plt.legend()
plt.savefig("loss")


# Generate predictions (probabilitites -- the output of the last layer)
y_pred = model.predict([X_test_c, X_test_n])
for i in range(20):
    category = ["A", "B", "C", "D"]
    key = np.argmax(y_pred[i])
   
    if key == 0:
        print("Pertenece al grupo A", " grupo verdadero es ", y_test.iloc[i].idxmax())
    if key == 1:
            print("Pertenece al grupo B", "        grupo verdadero es ",y_test.iloc[i].idxmax())

    if key == 2:
            print("Pertenece al grupo C","        grupo verdadero es ",y_test.iloc[i].idxmax())

    if key == 3:
            print("Pertenece al grupo D","        grupo verdadero es ",y_test.iloc[i].idxmax())

print("associate probabilities ")
print(y_pred[:20])

#0.5648
