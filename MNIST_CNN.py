from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import keras

(X_train,y_train),(X_test,y_test) = mnist.load_data()


X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = keras.Sequential()

model.add(keras.layers.Conv2D(64, 3, input_shape=(28,28,1), activation="relu"))
model.add(keras.layers.Conv2D(32, 3, activation="relu"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=1)
