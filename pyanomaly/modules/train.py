import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.deep_autoencoder import DeepAutoencoder

class Train:
    def __init__(self, input_file, output_file, epochs, lr, batch_size, loss, layers, h_activation, o_activation):
        self.input_file     = input_file
        self.df_input       = pd.read_csv(input_file)
        self.input_size     = len(self.df_input.columns)
        self.output_file    = output_file
        self.epochs         = epochs
        self.lr             = lr
        self.batch_size     = batch_size
        self.loss           = loss
        self.bias           = 1
        self.h_activation   = h_activation
        self.o_activation   = o_activation

        # Convert to list of strings then to list of ints
        self.layers  = [int(i) for i in layers.split(",")]

        self.config = {
            "input_size": self.input_size,
            "o_activation": self.o_activation,
            "h_activation": self.h_activation,
            "optimizer": {
                "name": "adam",
                "learning_rate": self.lr,
                "momentum": 0.0,
                "decay": 0.0
            },
            "encoding_layers": [],
            "decoding_layers": [],
            "epochs": self.epochs,
            "loss": self.loss,
            "bias": self.bias,
            "batch_size": self.batch_size
        }

    def execute(self):
        # Setup encoding layers
        encoding_layers = []
        for i in self.layers:
            encoding_layers.append({
                "size": i,
                "activation": self.h_activation,
                "bias": self.bias
            })

        self.config["encoding_layers"] = encoding_layers

        # Setup decoding layers
        decoding_layers = []
        for i in list(reversed(self.layers)):
            decoding_layers.append({
                "size": i,
                "activation": self.h_activation,
                "bias": self.bias
            })

        self.config["decoding_layers"] = decoding_layers

        self.autoencoder    = DeepAutoencoder(self.config)
        self.autoencoder.compile()
        self.autoencoder.summary()

        self.autoencoder.train(self.df_input)
