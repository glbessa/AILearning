import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, ReLU, Lambda
from tensorflow.keras.activations import sigmoid

class Autoencoder(Model):
    def __init__(self, name="Autoencoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.encoder = Sequential([
            Input(shape=(784,)),
            Dense(32),
            ReLU()
        ])
        self.decoder = Sequential([
            Input(shape=(32,)),
            Dense(784, activation=sigmoid)
        ])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded