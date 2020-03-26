# Pyanomaly: Autoencoder based anomaly detector

## Requirements

* Python 3.x
* Keras
* Pandas

## Trianing Mode

Train an autoencoder model to learn the identity function of some dataset.

### Syntax

```
python -m pyanomaly [-h] --mode training [--input INPUT] [--output OUTPUT]
                    [--epochs EPOCHS] [--lr LR] [--batch_size BATCH_SIZE]
                    [--loss {mse}] [--layers LAYERS]
                    [--output-activation {sigmoid,relu}]
                    [--hidden-activation {sigmoid,relu}]
```

## Normalize Mode

Normalize a dataset to scale from 0 to 1 using sklearn `MinMaxScaler`. Assumption is that the input csv file has a format of x1,x2,x3...y where `y` is the label for each datapoint. The program automatically drops the y column before normalizing.

Train an autoencoder model to learn the identity function of some dataset.

### Syntax

```
python -m pyanomaly [-h] --mode normalize [--input INPUT] [--output OUTPUT]
```
