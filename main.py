import os
import tensorflow as tf
from keras.datasets import cifar10
from tensorflow import keras
from tensorflow.python.keras import layers

# Load and normalize dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Define the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10),
])

# Define a Loss function and optimizer
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

# Train the network
model.fit(x_train, y_train, batch_size=4, epochs=10)

# Evaluate the model
model.evaluate(x_test, y_test)

# Save the model to the documents folder
save_path = os.path.expanduser("~/Documents/my_model.h5")
model.save(save_path)
