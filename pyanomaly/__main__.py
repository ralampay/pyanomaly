import sys
import argparse
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from modules.train import Train
from modules.normalize import Normalize
from modules.predict import Predict

def usage():
    print("python -m pyanomaly --mode [training] <options>")

parser  = argparse.ArgumentParser(description="Pyanomaly: Autoencoder based anomaly detector")
parser.add_argument("--mode", choices=["training", "normalize", "predict"], help="Mode to run. Possible values: training, normalize", required=True)
parser.add_argument("--input", help="Input csv file (Required if mode is training or normalize)")
parser.add_argument("--output", help="Output model (Required if mode is training) | Output normalized data (Required if mode is normalize)")
parser.add_argument("--epochs", help="Number of epochs (Required if mode is training)", type=int)
parser.add_argument("--lr", type=float, help="Learning rate")
parser.add_argument("--batch-size", type=int, help="Batch size")
parser.add_argument("--loss", choices=["mse", "binary_crossentropy"], help="Loss function")
parser.add_argument("--layers", help="Encoding layers")
parser.add_argument("--output-activation", choices=["sigmoid", "relu"], help="Output layer activation")
parser.add_argument("--hidden-activation", choices=["sigmoid", "relu"], help="Hidden layer activation")
parser.add_argument("--model", help="Model file (Required if mode is predict)")

args    = parser.parse_args()

if __name__ == '__main__':
    mode    = args.mode

    if mode == "normalize":
        input_file  = args.input
        output_file = args.output

        if not input_file:
            print("Error! Normalize mode requires --input value")
            parser.print_help()
            sys.exit(0)

        if not output_file:
            print("Error! Normalize mode requires --output value")
            parser.print_help()
            sys.exit(0)

        cmd = Normalize(input_file, output_file)
        cmd.execute()

    if mode == "predict":
        input_file      = args.input
        model_file      = args.model
        layers          = args.layers
        o_activation    = args.output_activation
        h_activation    = args.hidden_activation

        if not input_file:
            print("Error! Predict mode requires --input value")
            parser.print_help()
            sys.exit(0)

        if not model_file:
            print("Error! Predict mode requires --model value")
            parser.print_help()
            sys.exit(0)
        
        if not layers:
            print("Error! Predict mode requires --layers l1,l2,l3")
            parser.print_help()
            sys.exit(0)

        if not h_activation:
            print("Error! Predict mode requires --hidden-activation [sigmoid|relu]")
            parser.print_help()
            sys.exit(0)

        if not o_activation:
            print("Error! Predict mode requires --output-activation [sigmoid|relu]")
            parser.print_help()
            sys.exit(0)

        cmd = Predict(input_file, model_file, layers, h_activation, o_activation)
        cmd.execute()

    if mode == "training":
        input_file      = args.input
        output_file     = args.output
        epochs          = args.epochs
        lr              = args.lr
        batch_size      = args.batch_size
        loss            = args.loss
        layers          = args.layers
        o_activation    = args.output_activation
        h_activation    = args.hidden_activation

        if not input_file:
            print("Error! Training mode requires --input value")
            parser.print_help()
            sys.exit(0)

        if not output_file:
            print("Error! Training mode requires --output value")
            parser.print_help()
            sys.exit(0)

        if  not epochs:
            print("Error! Training mode requires --epochs value")
            parser.print_help()
            sys.exit(0)
        
        if not lr:
            print("Error! Training mode requires --lr value")
            parser.print_help()
            sys.exit(0)
        
        if not batch_size:
            print("Error! Training mode requires --batch-size value")
            parser.print_help()
            sys.exit(0)

        if not loss:
            print("Error! Training mode requires --loss value")
            parser.print_help()
            sys.exit(0)
        
        if not layers:
            print("Error! Training mode requires --layers l1,l2,l3")
            parser.print_help()
            sys.exit(0)

        if not h_activation:
            print("Error! Training mode requires --hidden-activation [sigmoid|relu]")
            parser.print_help()
            sys.exit(0)

        if not o_activation:
            print("Error! Training mode requires --output-activation [sigmoid|relu]")
            parser.print_help()
            sys.exit(0)

        cmd = Train(input_file, output_file, epochs, lr, batch_size, loss, layers, h_activation, o_activation)
        cmd.execute()
