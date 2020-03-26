import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.deep_autoencoder import DeepAutoencoder
from modules.utils import create_histogram

class Train:
    def __init__(self, input_file, output_file, epochs, lr, batch_size, loss, layers, h_activation, o_activation):
        self.input_file         = input_file
        self.df_input           = pd.read_csv(input_file)
        self.df_input_no_labels = self.df_input.drop(['y'], axis=1)
        self.input_size         = len(self.df_input_no_labels.columns)
        self.output_file        = output_file
        self.epochs             = epochs
        self.lr                 = lr
        self.batch_size         = batch_size
        self.loss               = loss
        self.bias               = 1
        self.h_activation       = h_activation
        self.o_activation       = o_activation
        self.column_names       = self.df_input_no_labels.columns
        self.df_shape           = self.df_input.shape
        self.num_dimensions     = self.input_size
        self.num_records        = self.df_shape[0]
        self.labels             = self.df_input['y'].values
        self.raw_values         = self.df_input_no_labels.values

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

        self.autoencoder = DeepAutoencoder(self.config)
        self.autoencoder.compile()
        self.autoencoder.summary()

        # Start training
        self.autoencoder.train(self.df_input_no_labels)

        # Compute for score
        self.reconstructed_data         = self.autoencoder.predict(self.df_input_no_labels.values)
        self.df_reconstructed_data      = pd.DataFrame(self.reconstructed_data, columns=self.column_names)
        self.reconstructed_td_errors    = np.power(self.df_reconstructed_data - self.df_input_no_labels, 2)
        self.mean_sq_errors             = np.mean(self.reconstructed_td_errors, axis=1)

        print("Shape of Input:", self.df_input_no_labels.shape)
        print("Shape of Reconstructed Data:", self.df_reconstructed_data.shape)
        print("Shape of Reconstructed Data errors:", self.reconstructed_td_errors.shape)
        print("Shape of Mean Squared Errors:", self.mean_sq_errors.shape)

        # Calculate the number of bins according to Freedman-Diaconis rule
        error_values    = self.mean_sq_errors.values
        bin_width       = 2 * iqr(error_values) / np.power(self.num_records, (1/3))
        num_bins        = (np.max(error_values) - np.min(error_values)) / bin_width

        self.hist, self.bins = create_histogram(error_values, num_bins=num_bins, step=bin_width)
        print("Bins:")
        print(self.bins)
        print("Num Bins:", len(self.bins))

        self.occurences = [float(x) for x in self.hist.tolist()]    # Convert to float data type
        print("Occurences:")
        print(self.occurences)

        print("Sum:")
        print(np.sum(self.occurences))

        # Plot
        labels = []
        for i in range(len(self.bins) - 1):
            labels.append(str(self.bins[i]) + "-" + str(self.bins[i + 1]))
        index = np.arange(len(labels))
        plt.bar(index, self.occurences)
        plt.xlabel('Error')
        plt.ylabel('Occurences')
        plt.xticks(index, labels, fontsize=5)
        plt.title('Histogram of Residual Errors')
        plt.grid(True)
        plt.show()

        print("Saving to file", self.output_file)
        self.autoencoder.save(self.output_file)
