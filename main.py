import os
import tensorflow as tf
from keras.datasets import cifar10
from tensorflow.python.keras import layers
import keras.utils

class ModuleWrapper(keras.layers.Layer):
    def __init__(self, module, **kwargs):
        super().__init__(**kwargs)
        self.module = module

    def call(self, inputs, training=False):
        return self.module(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "module": keras.utils.serialize_keras_object(self.module),
        })
        return config


# Load and normalize dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Define the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    # Add a convolutional neural network layer to detect shapes
    layers.Conv2D(64, (3, 3), activation="relu",
                   kernel_initializer="he_normal",
                   kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
    layers.MaxPooling2D((2, 2)),
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

# Save the model to the project files next to the .py file
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'my_model.tf')

# Save the model
model.save(save_path, save_format='tf')
