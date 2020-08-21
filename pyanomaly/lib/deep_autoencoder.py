import json
import pandas as pd
import numpy as np
import tensorflow as tf

from sys import argv
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.initializers import Constant
from datetime import datetime

class DeepAutoencoder:
    DEFAULT_CONFIG = {
        "input_size": 650,
        "o_activation": "sigmoid",
        "optimizer": {
            "name": "adam",
            "learning_rate": 0.001
        },
        "encoding_layers": [
            {
                "size": 600,
                "activation": "relu"
            }
        ],
        "decoding_layers": [
        ],
        "epochs": 1,
        "batch_size": 1,
        "loss": "mse",
        "bias": 0.1
    }

    def __init__(self, config = DEFAULT_CONFIG):
        self.config         = config
        self.o_activation   = self.config["o_activation"]
        self.optimizer      = self.config["optimizer"]
        self.epochs         = self.config["epochs"]
        self.batch_size     = self.config["batch_size"]
        self.input_size     = self.config["input_size"]
        self.bias           = self.config["bias"]

        # Encoder
        self.encoder = models.Sequential() 

        for i, c in enumerate(self.config["encoding_layers"][0:]):
            # First element refers to input
            if i == 0:
                self.encoder.add(
                    layers.Input(shape=(self.input_size,))
                )
            else:
                self.encoder.add(
                    layers.Dense(
                        c["size"],
                        activation=c["activation"],
                        bias_initializer=Constant(value=self.bias)
                    )
                )

        # Decoder
        self.decoder = models.Sequential()

        for i, c in enumerate(self.config["decoding_layers"][0:]):
            # First element refers to input (take last element of encoding_layers)
            if i == 0:
                self.decoder.add(
                    layers.Input(shape=(self.config["encoding_layers"][-1]["size"],))
                )
            else:
                self.decoder.add(
                    layers.Dense(
                        c["size"],
                        activation=c["activation"],
                        bias_initializer=Constant(value=self.bias)
                    )
                )

        # Autoencoder
        self.autoencoder = models.Sequential()
        self.autoencoder.add(
            layers.Input(shape=(self.input_size,))
        )

        self.autoencoder.add(self.encoder)
        self.autoencoder.add(self.decoder)



    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, z):
        return self.decoder.predict(z)

    def load_model(self, model_file):
        self.autoencoder = load_model(model_file)

    def predict(self, data):
        return self.autoencoder.predict(data)

    def compile(self):
        if self.config["optimizer"]["name"] == "sgd":
            self.autoencoder.compile(
                optimizer=optimizers.SGD(
                    lr=self.config["optimizer"]["learning_rate"]
                ),
                loss=self.config["loss"],
                metrics=['acc']
            )
        elif self.config["optimizer"]["name"] == "adam":
            self.autoencoder.compile(
                optimizer=optimizers.Adam(
                    lr=self.config["optimizer"]["learning_rate"]
                ),
                loss=self.config["loss"],
                metrics=['acc']
            )
        else:
            self.autoencoder.compile(
                optimizer=optimizers.Adam(
                    lr=self.config["optimizer"]["learning_rate"]
                ),
                loss=self.config["loss"],
                metrics=['acc']
            )

    def train(self, data):
        logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        self.autoencoder.fit(data, 
            data, 
            epochs=self.config["epochs"], 
            batch_size=self.config["batch_size"],
            callbacks=[tensorboard_callback]
        )

    def save(self, model_file):
        self.autoencoder.save(model_file)

    def summary(self):
        self.autoencoder.summary()
        self.encoder.summary()
        self.decoder.summary()
