import json
import pandas as pd
import numpy as np

from sys import argv
from keras import models
from keras.models import load_model
from keras import layers
from keras import optimizers
from keras.initializers import Constant

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

        # the actual autoencoder model
        self.autoencoder    = models.Sequential()

        # Input to Hidden
        for i, c in enumerate(self.config["encoding_layers"][1:]):
            if i == 0:
                self.autoencoder.add(
                    layers.Dense(
                        c["size"],
                        activation=c["activation"],
                        input_shape=(self.input_size,),
                        bias_initializer=Constant(value=self.bias)
                    )
                )
            else:
                self.autoencoder.add(
                    layers.Dense(
                        c["size"],
                        activation=c["activation"],
                        bias_initializer=Constant(value=self.bias)
                    )
                )
        
        # Hidden to Output
        for i, c in enumerate(self.config["decoding_layers"][1:(len(self.config["decoding_layers"]) - 1)]):
            self.autoencoder.add(
                layers.Dense(
                    c["size"],
                    activation=c["activation"],
                    bias_initializer=Constant(value=self.bias)
                )
            )

        # Output layer
        self.autoencoder.add(
            layers.Dense(
                self.config["input_size"],
                activation=self.config["o_activation"],
                bias_initializer=Constant(value=self.bias)
            )
        )

        # Encoder
        self.encoder = models.Sequential()

        for i, c in enumerate(self.config["encoding_layers"][1:]):
            if i == 0:
                self.encoder.add(
                    layers.Dense(
                        c["size"],
                        activation=c["activation"],
                        input_shape=(self.input_size,),
                        bias_initializer=Constant(value=self.bias)
                    )
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
        for i, c in enumerate(self.config["decoding_layers"][1:]):
            activation = c["activation"]

            if i == (len(self.config["decoding_layers"][1:]) - 1):
                activation = self.o_activation

            if i == 0:
                self.decoder.add(
                    layers.Dense(
                        c["size"],
                        activation=activation,
                        input_shape=(self.config["encoding_layers"][-1]["size"],),
                        bias_initializer=Constant(value=self.bias)
                    )
                )
            else:
                self.decoder.add(
                    layers.Dense(
                        c["size"],
                        activation=activation,
                        bias_initializer=Constant(value=self.bias)
                    )
                )


    def init_encoder(self):
        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i].set_weights(self.autoencoder.layers[i].get_weights())

    def init_decoder(self):
        for i, c in enumerate(self.config["decoding_layers"][1:]):
            self.decoder.layers[i].set_weights(self.autoencoder.layers[(i + len(self.encoder.layers))].get_weights())

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, z):
        return self.decoder.predict(z)

    def load_model(self, model_file):
        self.autoencoder = load_model(model_file)

        # Init the encoder and decoder
        self.init_encoder()
        self.init_decoder()

    def predict(self, data):
        return self.autoencoder.predict(data)

    def compile(self):
        if self.config["optimizer"]["name"] == "sgd":
            self.autoencoder.compile(
                optimizer=optimizers.SGD(
                    lr=self.config["optimizer"]["learning_rate"]
                ),
                loss=self.config["loss"]
            )
        elif self.config["optimizer"]["name"] == "adam":
            self.autoencoder.compile(
                optimizer=optimizers.Adam(
                    lr=self.config["optimizer"]["learning_rate"]
                ),
                loss=self.config["loss"]
            )
        else:
            self.autoencoder.compile(
                optimizer=optimizers.Adam(
                    lr=self.config["optimizer"]["learning_rate"]
                ),
                loss=self.config["loss"]
            )

    def train(self, data):
        self.autoencoder.fit(data, 
            data, 
            epochs=self.config["epochs"], 
            batch_size=self.config["batch_size"]
        )

        # Init the encoder and decoder
        self.init_encoder()
        self.init_decoder()

    def save(self, model_file):
        self.autoencoder.save(model_file)

    def summary(self):
        self.autoencoder.summary()
        self.encoder.summary()
        self.decoder.summary()
