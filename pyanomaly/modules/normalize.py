import pandas as pd
import sys
import os
from sklearn import preprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class Normalize:
    def __init__(self, input_file, output_file):
        self.input_file         = input_file
        self.output_file        = output_file
        self.df_input           = pd.read_csv(input_file)
        self.df_input_no_labels = self.df_input.drop(['y'], axis=1)
        self.column_names       = self.df_input_no_labels.columns
        self.df_shape           = self.df_input.shape
        self.num_dimensions     = self.df_shape[1] - 1
        self.num_records        = self.df_shape[0]
        self.labels             = self.df_input['y'].values
        self.raw_values         = self.df_input_no_labels.values

    def execute(self):
        print("Shape:", self.df_shape)
        print("Number of Dimensions:", self.num_dimensions)
        print("Number of Rows:", self.num_records)

        self.min_max_scaler     = preprocessing.MinMaxScaler()
        self.raw_values_scaled  = self.min_max_scaler.fit_transform(self.raw_values)
        self.df_normalized      = pd.DataFrame(self.raw_values_scaled, columns=self.column_names)
        self.df_normalized['y'] = self.labels

        print("Writing to file", self.output_file, "...")
        self.df_normalized.to_csv(self.output_file, sep=",", index=False)
