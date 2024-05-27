import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import optimizers, losses, metrics

from model import Autoencoder

if __name__ == "__main__":
    # Load MNIST dataset
    (x_train, _), (x_val, _) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    # Flatten images
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_val = x_val.reshape((len(x_val), np.prod(x_val.shape[1:])))

    # Setting up hyperparameters
    configs = {
        "batch_size": 10,
        "epochs": 10,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "loss_fn": "BinaryCrossentropy"
    }
    batch_size = configs["batch_size"]
    epochs = configs["epochs"]
    optimizer = getattr(tf.keras.optimizers, configs["optimizer"], 'Adam')
    #optimizer = optimizers.Adam
    learning_rate = configs["learning_rate"]
    loss_fn = getattr(tf.keras.losses, configs["loss_fn"], 'BinaryCrossentropy')
    #loss_fn = losses.BinaryCrossentropy

    # Create an instance of the model
    model = Autoencoder()
    model.build(input_shape=(None, 784))
    model.summary()
    model.compile(optimizer=optimizer(learning_rate=learning_rate), loss=loss_fn(), metrics=[
        metrics.BinaryAccuracy()
    ])

    model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, x_val), callbacks=[
        ModelCheckpoint("best.tf", save_format='tf', save_best_only=True),
        ModelCheckpoint("latest.tf", save_format='tf'),
        EarlyStopping(patience=3, monitor="val_loss"),
        TensorBoard(log_dir=os.path.join("logs", "autoencoder"))
    ])